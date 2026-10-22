# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 sparse MLA decode, following TileLang's DeepSeek MLA examples."""

import os

import torch

from vllm import envs
from vllm.tilelang_utils import T, tilelang

# ``VLLM_HY4_DECODE_NUM_SPLITS=1`` removes the split reduction from the
# kernel, making each decode query's output independent of the batch it
# runs in. ``VLLM_BATCH_INVARIANT=1`` implies that same single-split
# setting; the default None keeps the host heuristic below.
_FORCED_NUM_SPLITS = (
    int(value) if (value := os.getenv("VLLM_HY4_DECODE_NUM_SPLITS")) else None
)
if envs.VLLM_BATCH_INVARIANT and _FORCED_NUM_SPLITS is None:
    _FORCED_NUM_SPLITS = 1


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _decode_kernel(
    heads,
    topk,
    splits,
    scale,
    q_stride,
    kv_stride,
    index_stride,
    has_lengths,
    has_sink,
    use_async_copy,
):
    tokens = T.dynamic("tokens")
    rows = T.dynamic("rows")
    block_h = max(16, heads)
    block_k = 64
    blocks_per_split = T.ceildiv(T.ceildiv(topk, block_k), splits)
    scale_log2 = scale * 1.4426950408889634

    @T.prim_func
    def hy4_sparse_mla_decode(
        Q: T.StridedTensor((tokens, heads, 576), (q_stride, 576, 1), T.bfloat16),
        KV: T.StridedTensor((rows, 576), (kv_stride, 1), T.bfloat16),
        Indices: T.StridedTensor((tokens, topk), (index_stride, 1), T.int32),
        Lengths: T.Tensor((tokens,), T.int32),
        Sink: T.Tensor((heads,), T.float32),
        Partial: T.Tensor((tokens, splits, heads, 514), T.float32),
        Output: T.Tensor((tokens, heads, 512), T.bfloat16),
        Lse: T.Tensor((tokens, heads), T.float32),
    ):
        with T.Kernel(tokens, splits, threads=128) as (query, split):
            q = T.alloc_shared((block_h, 576), T.bfloat16)
            kv = T.alloc_shared((block_k, 512), T.bfloat16)
            rope = T.alloc_shared((block_k, 64), T.bfloat16)
            prob = T.alloc_shared((block_h, block_k), T.bfloat16)
            score = T.alloc_fragment((block_h, block_k), T.float32)
            acc = T.alloc_fragment((block_h, 512), T.float32)
            maximum = T.alloc_fragment((block_h,), T.float32)
            previous = T.alloc_fragment((block_h,), T.float32)
            alpha = T.alloc_fragment((block_h,), T.float32)
            denom = T.alloc_fragment((block_h,), T.float32)
            block_sum = T.alloc_fragment((block_h,), T.float32)
            valid = T.alloc_shared((block_k,), T.bool)
            ids = T.alloc_shared((block_k,), T.int32)
            T.copy(Q[query, :, :], q)
            T.fill(acc, 0)
            T.fill(denom, 0)
            T.fill(maximum, -1e30)
            for k in T.Pipelined(blocks_per_split, num_stages=1):
                for j in T.Parallel(block_k):
                    pos = (split * blocks_per_split + k) * block_k + j
                    ids[j] = T.if_then_else(pos < topk, Indices[query, pos], -1)
                    valid[j] = (pos < topk) & (ids[j] >= 0) & (ids[j] < rows)
                    if has_lengths:
                        valid[j] = valid[j] & (pos < Lengths[query])
                # Move aligned KV vectors directly into shared memory. Explicit
                # copies avoid register staging of the indirect, masked gather.
                if use_async_copy:
                    for c in T.serial(T.ceildiv(block_k * 64, 128)):
                        j = (c * 128 + T.get_thread_binding()) // 64
                        d = (c * 128 + T.get_thread_binding()) % 64
                        T.ptx_cp_async(
                            T.access_ptr(kv[j, d * 8], "w", 8),
                            T.access_ptr(KV[T.max(0, ids[j]), d * 8], "r", 8),
                            8,
                            valid[j],
                        )
                    for c in T.serial(T.ceildiv(block_k * 8, 128)):
                        j = (c * 128 + T.get_thread_binding()) // 8
                        d = (c * 128 + T.get_thread_binding()) % 8
                        T.ptx_cp_async(
                            T.access_ptr(rope[j, d * 8], "w", 8),
                            T.access_ptr(KV[T.max(0, ids[j]), 512 + d * 8], "r", 8),
                            8,
                            valid[j],
                        )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                else:
                    # Scalar loads also support an unaligned tensor base. A
                    # parallel copy may assume alignment from the row stride.
                    for c in T.serial(T.ceildiv(block_k * 512, 128)):
                        j = (c * 128 + T.get_thread_binding()) // 512
                        d = (c * 128 + T.get_thread_binding()) % 512
                        kv[j, d] = T.if_then_else(valid[j], KV[ids[j], d], 0)
                    for c in T.serial(T.ceildiv(block_k * 64, 128)):
                        j = (c * 128 + T.get_thread_binding()) // 64
                        d = (c * 128 + T.get_thread_binding()) % 64
                        rope[j, d] = T.if_then_else(valid[j], KV[ids[j], 512 + d], 0)
                T.clear(score)
                T.gemm(
                    q[:, :512],
                    kv,
                    score,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    q[:, 512:],
                    rope,
                    score,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for h, j in T.Parallel(block_h, block_k):
                    score[h, j] = T.if_then_else(
                        valid[j], score[h, j] * scale_log2, -T.infinity(T.float32)
                    )
                T.copy(maximum, previous)
                T.reduce_max(score, maximum, dim=1, clear=False)
                for h in T.Parallel(block_h):
                    alpha[h] = T.exp2(previous[h] - maximum[h])
                for h, j in T.Parallel(block_h, block_k):
                    score[h, j] = T.exp2(score[h, j] - maximum[h])
                T.reduce_sum(score, block_sum, dim=1)
                for h in T.Parallel(block_h):
                    denom[h] = denom[h] * alpha[h] + block_sum[h]
                for h, d in T.Parallel(block_h, 512):
                    acc[h, d] *= alpha[h]
                T.copy(score, prob)
                T.gemm(prob, kv, acc, policy=T.GemmWarpPolicy.FullRow)
            for h, d in T.Parallel(heads, 512):
                Partial[query, split, h, d] = acc[h, d]
            for h in T.Parallel(heads):
                Partial[query, split, h, 512] = maximum[h]
                Partial[query, split, h, 513] = denom[h]

        with T.Kernel(tokens, heads, threads=128) as (query, head):
            maximum = T.alloc_var(T.float32)
            denom = T.alloc_var(T.float32)
            merge_max = T.alloc_var(T.float32)
            normalization = T.alloc_var(T.float32)
            weight = T.alloc_var(T.float32)
            acc = T.alloc_fragment((512,), T.float32)
            maximum = -1e30
            denom = 0
            for split in T.serial(splits):
                maximum = T.max(maximum, Partial[query, split, head, 512])
            for split in T.serial(splits):
                denom += Partial[query, split, head, 513] * T.exp2(
                    Partial[query, split, head, 512] - maximum
                )
            if T.get_thread_binding() == 0:
                Lse[query, head] = T.if_then_else(
                    denom > 0,
                    (T.log2(denom) + maximum) * 0.6931471805599453,
                    T.infinity(T.float32),
                )
            merge_max = maximum
            normalization = T.if_then_else(denom > 0, 1 / denom, 0)
            if has_sink:
                # Add the sink once, after merging all KV partitions.
                if Sink[head] == T.infinity(T.float32):
                    normalization = 0
                else:
                    merge_max = T.max(maximum, Sink[head] * 1.4426950408889634)
                    normalization = T.if_then_else(
                        denom > 0,
                        1
                        / (
                            denom * T.exp2(maximum - merge_max)
                            + T.exp2(Sink[head] * 1.4426950408889634 - merge_max)
                        ),
                        0,
                    )
            T.clear(acc)
            for split in T.serial(splits):
                weight = T.exp2(Partial[query, split, head, 512] - merge_max)
                for d in T.Parallel(512):
                    acc[d] += Partial[query, split, head, d] * weight
            for d in T.Parallel(512):
                Output[query, head, d] = acc[d] * normalization

    return hy4_sparse_mla_decode


def sparse_mla_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend physical BF16 KV rows selected by each decode query's indices.

    Invalid indices and entries beyond topk_length contribute no mass. Returns
    BF16 output and natural-log LSE (excluding the sink), matching FlashMLA.
    """
    tokens, heads, dim = q.shape
    assert q.dtype == kv.dtype == torch.bfloat16
    assert dim == kv.shape[-1] == 576 and heads in (8, 16, 32)
    assert q.stride(1) == 576 and q.stride(2) == kv.stride(-1) == 1
    kv = kv.view(-1, 576)
    indices = indices.view(tokens, indices.shape[-1])
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    topk = indices.shape[1]
    if tokens == 0:
        return (
            q.new_empty((0, heads, 512)),
            torch.empty((0, heads), dtype=torch.float32, device=q.device),
        )
    assert topk > 0
    assert num_splits is None or num_splits > 0
    if num_splits is None:
        if _FORCED_NUM_SPLITS is not None:
            # Batch-invariant mode: a fixed split count keeps each query's
            # reduction order independent of the batch it runs in.
            assert _FORCED_NUM_SPLITS > 0
            num_splits = _FORCED_NUM_SPLITS
        else:
            sms = torch.cuda.get_device_properties(q.device).multi_processor_count
            num_splits = min(
                32, max(1, (sms + tokens - 1) // tokens), max(1, (topk + 63) // 64)
            )
    # Power-of-two reduction extent; split count depends only on host shapes.
    num_splits = 1 << (num_splits - 1).bit_length()
    out = torch.empty((tokens, heads, 512), dtype=q.dtype, device=q.device)
    lse = torch.empty((tokens, heads), dtype=torch.float32, device=q.device)
    partial = torch.empty(
        (tokens, num_splits, heads, 514), dtype=torch.float32, device=q.device
    )
    lengths = (
        topk_length if topk_length is not None else indices.as_strided((tokens,), (1,))
    )
    sink = attn_sink if attn_sink is not None else lse[0]
    _decode_kernel(
        heads,
        topk,
        num_splits,
        scale,
        q.stride(0),
        kv.stride(0),
        indices.stride(0),
        topk_length is not None,
        attn_sink is not None,
        kv.stride(0) % 8 == 0 and kv.storage_offset() % 8 == 0,
    )(q, kv, indices, lengths, sink, partial, out, lse)
    return out, lse
