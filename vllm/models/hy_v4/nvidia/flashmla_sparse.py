# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink-capable FlashMLA sparse backend for HY V4 (NVIDIA).

HY V4 adds a per-head learnable attention sink on top of sparse MLA. The
vendored FlashMLA kernels already accept an ``attn_sink`` argument, but vLLM's
shared ``FLASHMLA_SPARSE`` backend neither advertises sink support nor forwards
the tensor, so the bias would be silently dropped.

This module supplies the missing wiring inside the model package, mirroring how
`vllm.models.deepseek_v4.nvidia.flashinfer_sparse` hands ``sinks`` to the
FlashInfer sparse MLA kernels: subclass the platform backend, declare
`supports_sink`, and thread the sink into every kernel call.

The subclass intentionally keeps the inherited ``get_name()``
(``"FLASHMLA_SPARSE"``). Several shared code paths key off that exact string —
`_canonicalize_sparse_mla_kv_cache_dtype` promotes a quantized KV cache to
``fp8_ds_mla`` for it, and `FlashMLASparseImpl` asserts that layout — so a new
name would silently change KV cache behaviour.

On SM90 with BF16 KV, single-token decodes of 8/16/32 query heads may use the
TileLang split-KV kernel. Which kernel a step uses is decided per graph shape
when the metadata is built — the MiniMax M3 MSA pattern: the builder calls
`should_prepare_tilelang_decode_metadata`, records the answer in the metadata,
and the impl dispatches on that flag alone. The decision never depends on the
impl constructor, so a graph shape with a large batch keeps FlashMLA. A batch
is never split row-by-row between the two kernels: if any row is ineligible,
the whole batch runs FlashMLA.
"""

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLAPrefillMetadata,
    _use_dense_mha_prefill,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.hisparse.runtime import (
    HiSparsePrefillStagingPlan,
    build_hisparse_prefill_staging_plan,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


# TileLang's split-KV sparse decode is measured on these Hopper head counts.
_TILELANG_DECODE_HEADS = (8, 16, 32)
# Rows the TileLang kernel ever wins on: batch * query_len * query heads.
# H200, top-k 2048: 4096 rows still beats FlashMLA (1.2-1.4x), 6144 rows
# already loses (0.75x), so the measured crossover sits just below 4096.
_MAX_TILELANG_QUERY_HEAD_ROWS = 4096
_DEFAULT_TILELANG_DECODE_MIN_BATCH = 1


def _get_tilelang_decode_min_batch_size() -> int:
    """Smallest batch routed to the TileLang split-KV kernel.

    ``VLLM_HY4_TILELANG_DECODE_MIN_BATCH`` overrides the default, mirroring
    MiniMax M3's ``VLLM_MINIMAX_M3_MSA_CUTLASS_MIN_BATCH``.
    """
    value = os.getenv("VLLM_HY4_TILELANG_DECODE_MIN_BATCH")
    if value is None:
        return _DEFAULT_TILELANG_DECODE_MIN_BATCH
    return max(1, int(value))


def _tilelang_available(tilelang_available: bool | None) -> bool:
    if tilelang_available is not None:
        return tilelang_available
    from vllm.utils.import_utils import has_tilelang

    return has_tilelang()


def supports_tilelang_sparse_decode(
    *,
    num_q_heads: int,
    kv_cache_dtype: str,
    q_head_size: int | None,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    hisparse: bool = False,
    tilelang_available: bool | None = None,
) -> bool:
    """Return whether the static model geometry supports TileLang decode.

    Mirrors ``supports_cutlass_sparse_decode`` in MiniMax M3: pure model and
    platform properties, no batch shape. The batch shape is the second half of
    the gate, in `should_prepare_tilelang_decode_metadata`.

    HiSparse is excluded: its hot cache only holds the resident top-k rows, so
    the global-slot indices the TileLang kernel needs are not materialized.
    """
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
        and num_q_heads in _TILELANG_DECODE_HEADS
        and q_head_size == 576
        and kv_cache_dtype in ("auto", "bfloat16")
        and dcp_world_size == 1
        and pcp_world_size == 1
        and not hisparse
        and _tilelang_available(tilelang_available)
    )


def should_prepare_tilelang_decode_metadata(
    batch_size: int,
    decode_query_len: int,
    *,
    num_q_heads: int,
    kv_cache_dtype: str,
    q_head_size: int | None,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    hisparse: bool = False,
    tilelang_available: bool | None = None,
    min_batch_size: int | None = None,
) -> bool:
    """Return whether one graph shape can use the TileLang decode kernel.

    Mirrors ``should_prepare_decode_metadata`` in MiniMax M3: the builder calls
    this with the padded CUDA-graph shape, so the per-batch decision is part of
    the metadata and the same graph replays one kernel.
    """
    if min_batch_size is None:
        min_batch_size = _get_tilelang_decode_min_batch_size()
    return (
        supports_tilelang_sparse_decode(
            num_q_heads=num_q_heads,
            kv_cache_dtype=kv_cache_dtype,
            q_head_size=q_head_size,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hisparse=hisparse,
            tilelang_available=tilelang_available,
        )
        and decode_query_len == 1
        and batch_size >= min_batch_size
        and batch_size * decode_query_len * num_q_heads <= _MAX_TILELANG_QUERY_HEAD_ROWS
    )


@dataclass
class HYV4FlashMLASparseMetadata(FlashMLASparseMetadata):
    """Per-graph-shape BF16 kernel choice, made at metadata-build time.

    ``use_tilelang_decode`` selects one kernel for the whole batch. With it
    unset — the default for every batch the builder does not plan for TileLang
    — the batch stays on the FlashMLA kernel.

    ``batch_staging_plan`` is the HiSparse host-to-device staging remap for a
    mixed batch whose context pages are not all resident: it covers every row
    of the batch, so decode and prefill rows attend one gathered cache with a
    single kernel instead of being split.
    """

    use_tilelang_decode: bool = False
    batch_staging_plan: HiSparsePrefillStagingPlan | None = None


class HYV4FlashMLASparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    """Keep prompt chunks on the BF16 prefill path, including short extends."""

    metadata_cls = HYV4FlashMLASparseMetadata

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.use_fp8_kv_cache:
            self.require_uniform_decodes = False
        # Read through ``getattr`` so tests can build this builder over a
        # stubbed parent ``__init__``.
        mla_dims = getattr(self, "mla_dims", None)
        self.q_head_size = (
            None
            if mla_dims is None
            else mla_dims.kv_lora_rank + mla_dims.qk_rope_head_dim
        )
        self.kv_cache_dtype = getattr(
            getattr(self, "vllm_config", None), "cache_config", None
        )
        self.kv_cache_dtype = getattr(self.kv_cache_dtype, "cache_dtype", "auto")
        self.use_hisparse = getattr(self, "use_hisparse", False)
        self.tilelang_decode_min_batch_size = _get_tilelang_decode_min_batch_size()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        if self.use_fp8_kv_cache:
            return super().build(common_prefix_len, common_attn_metadata, fast_build)
        phases = common_attn_metadata.is_prefilling
        if phases is not None:
            # Graph buckets may include requests absent from the phase array.
            # Zero-query padding belongs with the decode prefix.
            num_reqs = common_attn_metadata.num_reqs
            if phases.numel() < num_reqs:
                padded = phases.new_zeros(num_reqs)
                padded[: phases.numel()] = phases
                common_attn_metadata = replace(
                    common_attn_metadata, is_prefilling=padded
                )
        return self._build_bf16_metadata(
            common_prefix_len, common_attn_metadata, fast_build
        )

    def _should_use_tilelang_decode(
        self,
        num_decodes: int,
        num_prefills: int,
        num_decode_tokens: int,
        num_actual_tokens: int,
        decode_query_len: int,
    ) -> bool:
        """Select TileLang from the padded decode batch shape.

        Mirrors MiniMax M3's metadata-time backend gate: the decision is made
        for the complete graph shape, so a larger batch never enters the
        split-KV kernel merely because the implementation was constructed for
        a small batch. Any batch that also carries prefill rows keeps FlashMLA
        for every row — rows are never split between the two kernels.
        """
        if num_prefills > 0 or num_decodes <= 0:
            return False
        if num_decode_tokens != num_actual_tokens:
            return False
        return should_prepare_tilelang_decode_metadata(
            num_decodes,
            decode_query_len,
            num_q_heads=getattr(self, "num_heads", 0),
            kv_cache_dtype=self.kv_cache_dtype,
            q_head_size=self.q_head_size,
            dcp_world_size=getattr(self, "dcp_world_size", 1),
            pcp_world_size=getattr(self, "pcp_world_size", 1),
            hisparse=self.use_hisparse,
            min_batch_size=self.tilelang_decode_min_batch_size,
        )

    def _build_batch_staging_plan(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_prefills: int,
        num_decode_tokens: int,
    ) -> HiSparsePrefillStagingPlan | None:
        """Gather a mixed batch's off-device context rows for one kernel.

        HiSparse keeps only the selected rows of a decode step in its hot
        cache. A batch that mixes decode rows with prompt chunks whose context
        pages are not resident needs one gathered cache for *every* row, so
        this plan covers all requests of the batch. The impl then maps all
        rows' top-k through that cache and runs a single kernel: decode rows
        are never handed to the hot cache and prefill rows to a staging buffer
        in the same step.

        Decode-only batches need no plan; they resolve through the resident
        hot cache in the impl. Mixed batches run eagerly (only uniform decode
        batches are captured), so rebuilding the device-side remap per step
        costs one gather plan, not one per layer.
        """
        if not getattr(self, "use_hisparse", False):
            return None
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        if num_prefills == 0 or num_decode_tokens >= num_actual_tokens:
            return None
        kv_cache_spec = getattr(self, "kv_cache_spec", None)
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if kv_cache_spec is None or seq_lens_cpu is None:
            # Without the host-side length bound the staging capacity cannot be
            # sized without a device sync; fall back to the impl's explicit
            # error rather than guessing.
            return None
        num_reqs = common_attn_metadata.num_reqs
        block_size = kv_cache_spec.block_size
        staged_blocks = (
            seq_lens_cpu[:num_reqs].to(torch.int64) + block_size - 1
        ) // block_size
        staging_block_capacity = int(staged_blocks.sum().item())
        if staging_block_capacity <= 0:
            return None
        return build_hisparse_prefill_staging_plan(
            common_attn_metadata.block_table_tensor[:num_reqs],
            common_attn_metadata.seq_lens[:num_reqs],
            block_size,
            staging_block_capacity,
        )

    def _build_bf16_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        req_id_per_token = self._build_req_id_per_token(common_attn_metadata)

        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
            treat_short_extends_as_decodes=False,
            require_uniform=self.require_uniform_decodes,
        )
        (
            prefill_query_start_loc,
            prefill_max_query_len,
            prefill_query_lens_cpu,
        ) = self._build_prefill_fields(common_attn_metadata, num_decodes, num_prefills)
        decode_max_query_len = 0
        if num_decodes:
            if common_attn_metadata.max_logits_per_req is not None:
                decode_max_query_len = common_attn_metadata.max_logits_per_req
            elif num_prefills == 0:
                decode_max_query_len = common_attn_metadata.max_query_len
            else:
                decode_query_lens_cpu = torch.diff(
                    common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
                )
                decode_max_query_len = int(decode_query_lens_cpu.max().item())
        use_tilelang_decode = self._should_use_tilelang_decode(
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            decode_query_len=decode_max_query_len,
        )

        batch_staging_plan = self._build_batch_staging_plan(
            common_attn_metadata, num_prefills, num_decode_tokens
        )
        prefill_max_seq_len = 0
        prefill: SparseMLAPrefillMetadata | None = None
        if num_prefills > 0 and self._prefill_backend is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_max_seq_len = int(
                seq_lens_cpu[num_decodes : num_decodes + num_prefills].max().item()
            )
            block_table = common_attn_metadata.block_table_tensor[num_decodes:, ...]
            chunked_context = self._build_chunked_context_fields(
                common_attn_metadata,
                num_decodes,
                num_prefills,
                prefill_query_lens_cpu,
            )
            staging_plan = None
            if self.vllm_config.attention_config.hisparse_config is not None:
                prefill_seq_lens_cpu = seq_lens_cpu[
                    num_decodes : num_decodes + num_prefills
                ]
                staging_block_capacity = int(
                    (
                        (prefill_seq_lens_cpu + self.kv_cache_spec.block_size - 1)
                        // self.kv_cache_spec.block_size
                    ).sum()
                )
                staging_plan = build_hisparse_prefill_staging_plan(
                    block_table,
                    common_attn_metadata.seq_lens[num_decodes:],
                    self.kv_cache_spec.block_size,
                    staging_block_capacity,
                )
                if chunked_context is not None:
                    block_table = staging_plan.block_table
            prefill = SparseMLAPrefillMetadata(
                block_table=block_table,
                query_start_loc=prefill_query_start_loc,
                max_query_len=prefill_max_query_len,
                query_lens_cpu=prefill_query_lens_cpu,
                chunked_context=chunked_context,
                q_data_type=self.model_config.dtype,
                output_dtype=self.model_config.dtype,
                prefill_backend=self._prefill_backend,
                use_dense_mha=(
                    _use_dense_mha_prefill(
                        self.vllm_config, prefill_max_seq_len, self.topk_tokens
                    )
                    # Dense MHA cannot be used under PCP+DCP.
                    and not (self.use_pcp and self.dcp_world_size > 1)
                ),
                topk_mask_workspace=self.topk_mask_workspace,
                host_staging_plan=staging_plan,
            )
            self._prefill_backend.prepare_metadata(prefill)

        return self.metadata_cls(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            decode_max_query_len=decode_max_query_len,
            use_tilelang_decode=use_tilelang_decode,
            batch_staging_plan=batch_staging_plan,
            prefill_max_seq_len=prefill_max_seq_len,
            prefill=prefill,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
        )


class HYV4FlashMLASparseImpl(FlashMLASparseImpl):
    """FlashMLA sparse impl that applies HY V4's per-head learnable sink.

    The sink enters as the ``sinks`` impl kwarg of
    `vllm.model_executor.layers.attention.MLAAttention` and is consumed by the
    FlashMLA kernels, which fold it into the softmax denominator:
    ``out *= exp(lse) / (exp(lse) + exp(sink))``.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        # ``SparseMLACommonImpl`` takes explicit keyword arguments only, so the
        # sink has to be removed before the base classes see``mla_args``.
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        self._validate_sinks(sinks, num_heads)
        self.sinks = sinks

    @staticmethod
    def _validate_sinks(sinks: torch.Tensor | None, num_heads: int) -> None:
        """Reject sink tensors the FlashMLA kernels cannot consume.

        Args:
            sinks: Candidate sink tensor, or None when the layer has no sink.
            num_heads: Local (TP-sharded) query head count of this layer.

        Raises:
            ValueError: If the dtype is not float32 or the shape is not
                ``(num_heads,)``.

        """
        if sinks is None:
            return
        if sinks.dtype != torch.float32:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have dtype "
                f"torch.float32, but got {sinks.dtype}."
            )
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have shape "
                f"({num_heads},), but got {tuple(sinks.shape)}."
            )

    def _sinks_for_query(
        self,
        q: torch.Tensor,
        head_dim: int,
        kernel_heads: int,
    ) -> torch.Tensor | None:
        """Return the sink laid out for the kernel's query head count.

        Args:
            q: Query tensor, before any head padding.
            head_dim: Axis of ``q`` holding the query heads.
            kernel_heads: Head count the kernel is invoked with, which may
                exceed the query head count because of padding.

        Returns:
            The sink tensor padded to ``kernel_heads`` with ``-inf`` (a no-op
            sink) for the padded lanes, or None when the layer has no sink.

        Raises:
            ValueError: If the sink and query head layouts disagree, or if they
                live on different devices.

        """
        sinks = self.sinks
        if sinks is None:
            return None

        query_heads = q.shape[head_dim]
        if sinks.shape[0] != query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sink head count must match the "
                f"runtime query layout: sinks={sinks.shape[0]}, "
                f"query_heads={query_heads}. The sink must use the same "
                "unpadded head layout as the query."
            )
        if sinks.device != q.device:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks and query must be on the "
                f"same device, but got sinks={sinks.device}, query={q.device}."
            )
        if kernel_heads < query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse kernel head count cannot be smaller than "
                f"the runtime query layout: query_heads={query_heads}, "
                f"kernel_heads={kernel_heads}."
            )
        if kernel_heads == query_heads:
            return sinks

        # Mirror the query padding the kernels require. Reading ``sinks`` here
        # (rather than caching a padded copy at construction time) keeps the
        # values correct for weights loaded after the module is built, and the
        # allocation plus copy are captured in the CUDA graph.
        padded_sinks = sinks.new_full((kernel_heads,), float("-inf"))
        padded_sinks[:query_heads] = sinks
        return padded_sinks

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads
        attn_sink = self._sinks_for_query(q, head_dim=2, kernel_heads=padded_num_heads)

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
            attn_sink=attn_sink,
        )

        # Slice output and lse back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]
            lse = lse[:, :actual_num_heads, :]

        return out, lse

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        actual_num_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one BF16 sparse kernel over the whole batch.

        The batch is never split row-by-row: whatever kernel
        ``attn_metadata.use_tilelang_decode`` selects handles every row, and a
        HiSparse mixed batch gathers its non-resident context rows once so the
        same single-kernel rule holds there too.
        """
        index_group = self.index_group
        uses_host_cache = isinstance(index_group, HiSparseMLAIndexGroup)
        num_rows = topk_indices.shape[0]
        # req_id_per_token covers the whole batch; slice it to the MQA tokens
        # (q may exclude prefill tokens routed to dense MHA).
        req_id_per_token = attn_metadata.req_id_per_token[:num_rows]
        block_table = attn_metadata.block_table
        block_stride_rows: int | None = None
        kv_rows = kv_c_and_k_pe_cache

        if uses_host_cache:
            assert isinstance(index_group, HiSparseMLAIndexGroup)
            cache = index_group.cache(self.index_group_index)
            decode_only = (
                attn_metadata.num_decode_tokens
                == attn_metadata.num_actual_tokens
                == num_rows
            )
            if cache.all_context_pages_resident or decode_only:
                # Every selected row is (or can be swapped) into the hot cache,
                # so the whole batch addresses it directly.
                physical_topk, lengths = index_group.convert_logical_to_physical_topk(
                    self.index_group_index,
                    topk_indices,
                    attn_metadata,
                    block_stride_rows=None,
                    return_valid_counts=True,
                )
                return self._bf16_attention(
                    q,
                    index_group.physical_kv_cache(self.index_group_index),
                    physical_topk,
                    lengths,
                    attn_metadata,
                    actual_num_heads,
                )
            # Mixed batch with off-device context pages: stage every row of the
            # batch into one cache instead of reading decode rows from the hot
            # cache and prompt rows from a staging buffer.
            plan = getattr(attn_metadata, "batch_staging_plan", None)
            if plan is None:
                raise RuntimeError(
                    "HiSparse mixed-batch BF16 attention requires a whole-batch "
                    "staging plan, but the attention metadata carries none. "
                    "Build the metadata with HYV4FlashMLASparseMetadataBuilder so "
                    "batch_staging_plan is populated."
                )
            resident = cache.view.cache if cache.view is not None else None
            state_indices = cache.runtime.request_state_indices
            if resident is not None and state_indices is not None:
                # Requests with a GPU-resident copy are gathered device-to-device;
                # rows of this batch keep the batch's request order.
                plan.ensure_gpu_sources(
                    cache.block_table,
                    state_indices[: plan.block_table.shape[0]],
                    cache.view.block_size,
                )
            staged = cache.runtime.gather_prefill_cache(
                kv_c_and_k_pe_cache,
                plan,
                resident_cache=resident,
            )
            kv_rows = staged.view(-1, staged.shape[-1])
            block_table = plan.block_table
        elif kv_c_and_k_pe_cache.dim() == 2:
            # Already a flat [row, head_dim] view: rows are addressed directly.
            kv_rows = kv_c_and_k_pe_cache
        else:
            kv_rows, block_stride_rows = flat_kv_row_view(
                kv_c_and_k_pe_cache, attn_metadata.block_size
            )

        # Convert per-request indices to global slots (decode) or staging
        # offsets (mixed batch); the kernel choice never changes the mapping.
        decode_only = (
            attn_metadata.num_decode_tokens
            == attn_metadata.num_actual_tokens
            == num_rows
        )
        if not uses_host_cache and decode_only and self.index_group is not None:
            topk_indices, topk_length = self._convert_logical_to_physical_topk(
                topk_indices,
                attn_metadata,
                block_stride_rows=block_stride_rows,
                return_valid_counts=True,
            )
        else:
            topk_indices, topk_length = triton_convert_req_index_to_global_index(
                req_id_per_token,
                block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )

        return self._bf16_attention(
            q,
            kv_rows,
            topk_indices,
            topk_length,
            attn_metadata,
            actual_num_heads,
        )

    def _bf16_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor | None,
        attn_metadata: FlashMLASparseMetadata,
        actual_num_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not getattr(attn_metadata, "use_tilelang_decode", False):
            return self._bf16_flash_mla_kernel(
                q, kv, indices, lengths, actual_num_heads
            )
        from vllm.models.hy_v4.nvidia.sparse_mla_decode import sparse_mla_decode

        return sparse_mla_decode(
            q[:, :actual_num_heads],
            kv,
            indices,
            self.softmax_scale,
            lengths,
            self.sinks,
        )

    @staticmethod
    def _compute_bf16_padded_heads(num_heads: int) -> int:
        if current_platform.is_device_capability_family(90) and num_heads in (
            8,
            16,
            32,
        ):
            return num_heads
        return round_up(num_heads, 64)

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # Pad from q's head count, not self.num_heads: under DCP the heads are
        # all-gathered before this.
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        padded_num_heads = self._compute_bf16_padded_heads(actual_num_heads)
        # q_concat_buffer may already include the kernel-required padding.
        # The sink remains defined only for the real query heads.
        attn_sink = self._sinks_for_query(
            q[:, :actual_num_heads], head_dim=1, kernel_heads=padded_num_heads
        )

        if q.shape[1] < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for BF16 sparse prefill kernel"
            )
            # Zero (not new_empty) the padded lanes: topk_indices is shared by
            # all heads, so the kernel reduces across the head group and NaNs
            # from uninitialized memory would leak into the real heads.
            if out is None:
                q_padded = q.new_zeros((q.shape[0], padded_num_heads, q.shape[2]))
            else:
                *_, q_padded, _, _ = current_workspace_manager().get_simultaneous(
                    *self.workspace_specs
                )
                q_padded = q_padded[:num_tokens]
                q_padded[:, actual_num_heads:].zero_()
            q_padded[:, :actual_num_heads, :] = q[:, :actual_num_heads]
            q = q_padded
        else:
            q = q[:, :padded_num_heads]

        if out is not None:
            out = out[:, :padded_num_heads]

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output, _, lse = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
        )

        output = output[:, :actual_num_heads, :]
        lse = lse[:, :actual_num_heads]
        return output, lse


class HYV4FlashMLASparseBackend(FlashMLASparseBackend):
    """``FLASHMLA_SPARSE`` with attention-sink support for HY V4.

    Keeps the parent's name and metadata; the builder distinguishes prompt
    chunks from generated tokens, and the impl applies the sink. See the module
    docstring for why the name is reused.
    """

    @staticmethod
    def get_builder_cls() -> type[HYV4FlashMLASparseMetadataBuilder]:
        return HYV4FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[HYV4FlashMLASparseImpl]:
        return HYV4FlashMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
