# Copyright (c) OpenMMLab. All rights reserved.

import functools
from dataclasses import dataclass
from typing import Literal

import torch

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.utils import get_logger

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata

logger = get_logger('lmdeploy')

use_fa3 = False
try:
    # Now flash-attention only support FA3 for sm90a && cuda >= 12.3
    if (torch.cuda.get_device_capability()[0] == 9) and (torch.version.cuda >= '12.3'):
        import flash_attn_interface  # noqa: F401
        assert torch.ops.flash_attn_3 is not None
        use_fa3 = True
except Exception:
    logger.debug('For higher performance, please install FlashAttention-3 '
                 'https://github.com/Dao-AILab/flash-attention')


@dataclass
class TritonAttentionMetadata(AttentionMetadata):
    """Triton attention metadata."""
    is_decoding: bool
    block_offsets: torch.Tensor
    q_start_loc: torch.Tensor = None
    q_seqlens: torch.Tensor = None
    kv_start_loc: torch.Tensor = None
    kv_seqlens: torch.Tensor = None
    fill_seqlens: torch.Tensor = None
    quant_policy: Literal[0, 4, 8] = 0
    kv_flatten_size: int = None
    # flash mla
    tile_scheduler_metadata: torch.Tensor = None
    num_splits: torch.Tensor = None
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None


def _cdiv(a, b):
    """Perform div up."""
    return (a + b - 1) // b


class TritonAttentionImpl(AttentionImpl[TritonAttentionMetadata]):
    """Triton attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = None,
        causal: bool = True,
        block_sparse_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            causal=causal,
            **kwargs,
        )
        assert not (alibi and not causal)

        from lmdeploy.pytorch.kernels.cuda import (alibi_paged_attention_fwd, fill_kv_cache, flash_attention_fwd,
                                                   flatten_kv_cache, paged_attention_fwd, paged_attention_fwd_causal)

        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = paged_attention_fwd
        self.paged_attention_fwd_causal = paged_attention_fwd_causal
        self.alibi_paged_attention_fwd = alibi_paged_attention_fwd
        self.flatten_kv_cache = flatten_kv_cache
        self.flash_attention_fwd = flash_attention_fwd

        # for alibi attention
        world_size, rank = get_tp_world_rank()
        self.alibi_head_offset = self.num_heads * rank
        self.alibi_num_heads = self.num_heads * world_size
        self.block_sparse_size = block_sparse_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        """forward."""
        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        fill_q_start_loc = q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        fill_seqlens = q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        if attn_metadata.is_decoding:
            max_q_seqlen = self.block_sparse_size
        else:
            max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        fill_max_q_seqlen = max_q_seqlen
        if attn_metadata.fill_seqlens is not None:
            fill_seqlens = attn_metadata.fill_seqlens
            fill_max_q_seqlen = key.numel() // (key.size(-1) * key.size(-2))
            fill_q_start_loc = fill_seqlens.cumsum(0) - fill_seqlens

        # fill kv cache
        if key is not None and value is not None:
            self.fill_kv_cache(
                key,
                value,
                k_cache,
                v_cache,
                fill_q_start_loc,
                fill_seqlens,
                kv_seq_length=kv_seqlens,
                max_q_seq_length=fill_max_q_seqlen,
                block_offsets=block_offsets,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
            )

        q_shape = query.shape
        o_shape = q_shape[:-1] + (self.v_head_size, )
        attn_output = query.new_empty(o_shape)
        is_decoding = attn_metadata.is_decoding

        if self.alibi:
            self.alibi_paged_attention_fwd(
                query,
                k_cache,
                v_cache,
                attn_output,
                block_offsets,
                b_start_loc=q_start_loc,
                b_seq_len=q_seqlens,
                b_kv_seq_len=kv_seqlens,
                max_input_len=max_q_seqlen,
                head_offset=self.alibi_head_offset,
                num_heads=self.alibi_num_heads,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
            )
            return attn_output

        if is_decoding:
            # def eager_attention_forward_causal():
            #     """Eager attention forward as reference implementation for paged_attention_fwd.
                
            #     Supports multiple tokens per request in decoding phase.
            #     Within a block (same request), uses causal (unidirectional) attention.
            #     All tokens can attend to previous context tokens.
            #     """
            #     import math
            #     # query shape: [num_tokens, num_heads, head_dim]
            #     # k_cache, v_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
                
            #     num_tokens = query.size(0)
            #     num_heads = query.size(1)
            #     head_dim = query.size(2)
            #     num_kv_heads = k_cache.size(2)
            #     block_size = k_cache.size(1)
                
            #     # Group query attention: repeat kv heads to match query heads
            #     num_queries_per_kv = num_heads // num_kv_heads
                
            #     # Determine request boundaries from q_start_loc and q_seqlens
            #     # q_start_loc: starting position of each request in the batch
            #     # q_seqlens: number of query tokens for each request
            #     num_requests = len(q_seqlens)
                
            #     outputs = []
                
            #     # Process each request
            #     for req_idx in range(num_requests):
            #         req_start = q_start_loc[req_idx].item()
            #         req_num_query_tokens = q_seqlens[req_idx].item()
            #         req_end = req_start + req_num_query_tokens
                    
            #         # Extract query tokens for this request
            #         q = query[req_start:req_end]  # [req_num_query_tokens, num_heads, head_dim]
                    
            #         # Get KV sequence length for this request
            #         kv_seq_len = kv_seqlens[req_idx].item()
                    
            #         # Extract K and V from paged cache for this request
            #         req_block_offsets = block_offsets[req_idx]  # [num_blocks_per_seq]
            #         num_blocks = (kv_seq_len + block_size - 1) // block_size
                    
            #         # Gather K and V
            #         k_list = []
            #         v_list = []
            #         for block_idx in range(num_blocks):
            #             block_id = req_block_offsets[block_idx].item()
            #             k_block = k_cache[block_id]  # [block_size, num_kv_heads, head_dim]
            #             v_block = v_cache[block_id]  # [block_size, num_kv_heads, head_dim or v_head_dim]
            #             k_list.append(k_block)
            #             v_list.append(v_block)
                    
            #         k = torch.cat(k_list, dim=0)[:kv_seq_len]  # [kv_seq_len, num_kv_heads, head_dim]
            #         v = torch.cat(v_list, dim=0)[:kv_seq_len]  # [kv_seq_len, num_kv_heads, v_head_dim]
                    
            #         # Reshape for group query attention
            #         k = k.unsqueeze(2).expand(-1, -1, num_queries_per_kv, -1)
            #         k = k.reshape(kv_seq_len, num_heads, head_dim)
                    
            #         v = v.unsqueeze(2).expand(-1, -1, num_queries_per_kv, -1)
            #         v = v.reshape(kv_seq_len, num_heads, self.v_head_size)
                    
            #         # Transpose for batched matmul
            #         # q: [req_num_query_tokens, num_heads, head_dim] -> [num_heads, req_num_query_tokens, head_dim]
            #         # k: [kv_seq_len, num_heads, head_dim] -> [num_heads, head_dim, kv_seq_len]
            #         # v: [kv_seq_len, num_heads, v_head_dim] -> [num_heads, kv_seq_len, v_head_dim]
            #         q_t = q.transpose(0, 1)  # [num_heads, req_num_query_tokens, head_dim]
            #         k_t = k.permute(1, 2, 0)  # [num_heads, head_dim, kv_seq_len]
            #         v_t = v.transpose(0, 1)  # [num_heads, kv_seq_len, v_head_dim]
                    
            #         # Compute attention scores: Q @ K^T
            #         scale = self.scale if self.scale is not None else (1.0 / math.sqrt(head_dim))
            #         scores = torch.matmul(q_t, k_t) * scale  # [num_heads, req_num_query_tokens, kv_seq_len]
                    
            #         # Create attention mask for causal attention (unidirectional)
            #         # Current block: last req_num_query_tokens in the kv_seq_len
            #         # Previous tokens: first (kv_seq_len - req_num_query_tokens) tokens
            #         prev_context_len = kv_seq_len - req_num_query_tokens
                    
            #         # Create causal mask
            #         # Initialize with -inf to mask all positions
            #         mask = torch.full((req_num_query_tokens, kv_seq_len), 
            #                          float('-inf'), dtype=scores.dtype, device=scores.device)
                    
            #         # Previous context: all query tokens can attend to all previous context tokens
            #         if prev_context_len > 0:
            #             mask[:, :prev_context_len] = 0  # No masking for previous context
                    
            #         # Current block: causal (unidirectional) attention
            #         # Query token i can only attend to tokens 0 to i in the current block
            #         for q_idx in range(req_num_query_tokens):
            #             # Token q_idx can attend to tokens 0 to q_idx (inclusive) in current block
            #             mask[q_idx, prev_context_len:prev_context_len + q_idx + 1] = 0
                    
            #         # Apply mask to scores
            #         # Expand mask to match scores shape: [num_heads, req_num_query_tokens, kv_seq_len]
            #         mask = mask.unsqueeze(0)  # [1, req_num_query_tokens, kv_seq_len]
            #         scores = scores + mask  # Broadcasting over num_heads dimension
                    
            #         # Softmax
            #         attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, req_num_query_tokens, kv_seq_len]
                    
            #         # Compute output: attn_weights @ V
            #         output = torch.matmul(attn_weights, v_t)  # [num_heads, req_num_query_tokens, v_head_dim]
            #         output = output.transpose(0, 1)  # [req_num_query_tokens, num_heads, v_head_dim]
            #         outputs.append(output)
                
            #     # Concatenate outputs from all requests
            #     eager_output = torch.cat(outputs, dim=0)  # [num_tokens, num_heads, v_head_dim]
            #     return eager_output
            
            # def eager_attention_forward_noncausal():
            #     """Eager attention forward as reference implementation for paged_attention_fwd.
                
            #     Supports multiple tokens per request in decoding phase.
            #     Within a block (same request), uses bidirectional attention.
            #     Across different blocks/requests, uses causal attention.
            #     """
            #     import math
            #     # query shape: [num_tokens, num_heads, head_dim]
            #     # k_cache, v_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
                
            #     num_tokens = query.size(0)
            #     num_heads = query.size(1)
            #     head_dim = query.size(2)
            #     num_kv_heads = k_cache.size(2)
            #     block_size = k_cache.size(1)
                
            #     # Group query attention: repeat kv heads to match query heads
            #     num_queries_per_kv = num_heads // num_kv_heads
                
            #     # Determine request boundaries from q_start_loc and q_seqlens
            #     # q_start_loc: starting position of each request in the batch
            #     # q_seqlens: number of query tokens for each request
            #     num_requests = len(q_seqlens)
                
            #     outputs = []
                
            #     # Process each request
            #     for req_idx in range(num_requests):
            #         req_start = q_start_loc[req_idx].item()
            #         req_num_query_tokens = q_seqlens[req_idx].item()
            #         req_end = req_start + req_num_query_tokens
                    
            #         # Extract query tokens for this request
            #         q = query[req_start:req_end]  # [req_num_query_tokens, num_heads, head_dim]
                    
            #         # Get KV sequence length for this request
            #         kv_seq_len = kv_seqlens[req_idx].item()
                    
            #         # Extract K and V from paged cache for this request
            #         req_block_offsets = block_offsets[req_idx]  # [num_blocks_per_seq]
            #         num_blocks = (kv_seq_len + block_size - 1) // block_size
                    
            #         # Gather K and V
            #         k_list = []
            #         v_list = []
            #         for block_idx in range(num_blocks):
            #             block_id = req_block_offsets[block_idx].item()
            #             k_block = k_cache[block_id]  # [block_size, num_kv_heads, head_dim]
            #             v_block = v_cache[block_id]  # [block_size, num_kv_heads, head_dim or v_head_dim]
            #             k_list.append(k_block)
            #             v_list.append(v_block)
                    
            #         k = torch.cat(k_list, dim=0)[:kv_seq_len]  # [kv_seq_len, num_kv_heads, head_dim]
            #         v = torch.cat(v_list, dim=0)[:kv_seq_len]  # [kv_seq_len, num_kv_heads, v_head_dim]
                    
            #         # Reshape for group query attention
            #         k = k.unsqueeze(2).expand(-1, -1, num_queries_per_kv, -1)
            #         k = k.reshape(kv_seq_len, num_heads, head_dim)
                    
            #         v = v.unsqueeze(2).expand(-1, -1, num_queries_per_kv, -1)
            #         v = v.reshape(kv_seq_len, num_heads, self.v_head_size)
                    
            #         # Transpose for batched matmul
            #         # q: [req_num_query_tokens, num_heads, head_dim] -> [num_heads, req_num_query_tokens, head_dim]
            #         # k: [kv_seq_len, num_heads, head_dim] -> [num_heads, head_dim, kv_seq_len]
            #         # v: [kv_seq_len, num_heads, v_head_dim] -> [num_heads, kv_seq_len, v_head_dim]
            #         q_t = q.transpose(0, 1)  # [num_heads, req_num_query_tokens, head_dim]
            #         k_t = k.permute(1, 2, 0)  # [num_heads, head_dim, kv_seq_len]
            #         v_t = v.transpose(0, 1)  # [num_heads, kv_seq_len, v_head_dim]
                    
            #         # Compute attention scores: Q @ K^T
            #         scale = self.scale if self.scale is not None else (1.0 / math.sqrt(head_dim))
            #         scores = torch.matmul(q_t, k_t) * scale  # [num_heads, req_num_query_tokens, kv_seq_len]
                    
            #         # Create attention mask for bidirectional attention within the current block
            #         # and causal attention for previous tokens
            #         # Current block: last req_num_query_tokens in the kv_seq_len
            #         # Previous tokens: first (kv_seq_len - req_num_query_tokens) tokens
            #         mask = torch.zeros((req_num_query_tokens, kv_seq_len), 
            #                           dtype=scores.dtype, device=scores.device)
                    
            #         # Previous context: causal attention from all query tokens
            #         prev_context_len = kv_seq_len - req_num_query_tokens
            #         if prev_context_len > 0:
            #             # All query tokens can attend to all previous context tokens
            #             mask[:, :prev_context_len] = 0  # No masking for previous context
                    
            #         # Current block: bidirectional attention
            #         # Each query token can attend to all tokens in the current block
            #         for q_idx in range(req_num_query_tokens):
            #             # Bidirectional attention within current block
            #             mask[q_idx, prev_context_len:] = 0  # No masking within block
                    
            #         # Apply mask to scores (mask with -inf for positions that should not be attended)
            #         # Since we want no masking (bidirectional), we don't need to apply mask
            #         # But if we want causal within block, we would create a causal mask
                    
            #         # For now, bidirectional within block means no additional masking needed
            #         # The mask is already all zeros which means no masking
                    
            #         # Softmax
            #         attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, req_num_query_tokens, kv_seq_len]
                    
            #         # Compute output: attn_weights @ V
            #         output = torch.matmul(attn_weights, v_t)  # [num_heads, req_num_query_tokens, v_head_dim]
            #         output = output.transpose(0, 1)  # [req_num_query_tokens, num_heads, v_head_dim]
            #         outputs.append(output)
                
            #     # Concatenate outputs from all requests
            #     eager_output = torch.cat(outputs, dim=0)  # [num_tokens, num_heads, v_head_dim]
            #     return eager_output
            
            # attn_output_eager_causal = eager_attention_forward_causal()
            # attn_output_eager_noncausal = eager_attention_forward_noncausal()
            
            self.paged_attention_fwd_causal(
                query,
                k_cache,
                v_cache,
                attn_output,
                block_offsets,
                kv_seqlens=kv_seqlens,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                window_size=self.sliding_window,
                sm_scale=self.scale,
                logit_softcapping=self.logit_softcapping,
                sinks=learnable_sink,
            )
            # Validate correctness
            # similarity_causal = torch.cosine_similarity(
            #     attn_output_causal.view(-1),
            #     attn_output_eager_causal.view(-1),
            #     dim=0,
            #     eps=1e-6,
            # )
            # similarity_noncausal = torch.cosine_similarity(
            #     attn_output_causal.view(-1),
            #     attn_output_eager_noncausal.view(-1),
            #     dim=0,
            #     eps=1e-6,
            # )
            # print(f'TritonAttentionImpl decoding causal similarity: {similarity_causal.item()}, noncausal similarity: {similarity_noncausal.item()}')
        else:
            BLOCK_BS = k_cache.size(1)
            # pad one more block to avoid invalid kv visit
            out_size = (_cdiv(kv_flatten_size, BLOCK_BS) * BLOCK_BS + BLOCK_BS)
            flatten_k, flatten_v = self.flatten_kv_cache(
                k_cache,
                v_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=out_size,
                out_dtype=query.dtype,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
            )
            self.flash_attention_fwd(
                query,
                flatten_k,
                flatten_v,
                attn_output,
                q_start_loc=q_start_loc,
                q_seqlens=q_seqlens,
                kv_start_loc=kv_start_loc,
                kv_seqlens=kv_seqlens,
                max_seqlen=max_q_seqlen,
                window_size=self.sliding_window,
                sm_scale=self.scale,
                logit_softcapping=self.logit_softcapping,
                sinks=learnable_sink,
                causal=self.causal,
                block_sparse_size=self.block_sparse_size,
            )

        return attn_output


@functools.lru_cache
def use_fa3_warning():
    if use_fa3:
        return True
    logger.warning('For higher performance, please install FlashAttention-3 '
                   'https://github.com/Dao-AILab/flash-attention')
    return False


class FlashMLAImpl(TritonAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = None,
        causal: bool = True,
        **kwargs,
    ):
        assert sliding_window is None, 'sliding window not supported for FlashMLA'
        assert alibi is False, 'alibi not supported for FlashMLA'
        assert logit_softcapping is None, 'logit_softcapping not supported for FlashMLA'
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            causal=causal,
            **kwargs,
        )

        from lmdeploy.pytorch.kernels.cuda import flash_mla_fwd
        self.flash_mla_fwd = flash_mla_fwd
        assert num_kv_heads == 1, 'MLA requires num kv heads equal to 1'
        use_fa3_warning()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        """forward."""

        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        fill_q_start_loc = q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        fill_seqlens = q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        if attn_metadata.is_decoding:
            max_q_seqlen = 1
        else:
            max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        fill_max_q_seqlen = max_q_seqlen
        if attn_metadata.fill_seqlens is not None:
            fill_seqlens = attn_metadata.fill_seqlens
            fill_max_q_seqlen = key.numel() // (key.size(-1) * key.size(-2))
            fill_q_start_loc = fill_seqlens.cumsum(0) - fill_seqlens

        # fill kv cache
        if key is not None and value is not None:
            self.fill_kv_cache(
                key,
                value,
                k_cache,
                v_cache,
                fill_q_start_loc,
                fill_seqlens,
                kv_seq_length=kv_seqlens,
                max_q_seq_length=fill_max_q_seqlen,
                block_offsets=block_offsets,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
            )

        q_shape = query.shape
        o_shape = q_shape[:-1] + (self.v_head_size, )

        is_decoding = attn_metadata.is_decoding
        if is_decoding:
            query = query.unsqueeze(1)
            if kv_seqlens.dtype == torch.int64:
                kv_seqlens = kv_seqlens.to(torch.int32)
            attn_output = self.flash_mla_fwd(query,
                                             k_cache=k_cache,
                                             block_table=block_offsets,
                                             cache_seqlens=kv_seqlens,
                                             head_dim_v=self.v_head_size,
                                             softmax_scale=self.scale,
                                             tile_scheduler_metadata=attn_metadata.tile_scheduler_metadata,
                                             num_splits=attn_metadata.num_splits,
                                             causal=True)
        else:
            BLOCK_BS = k_cache.size(1)
            # pad one more block to avoid invalid kv visit
            out_size = (_cdiv(kv_flatten_size, BLOCK_BS) * BLOCK_BS + BLOCK_BS)
            flatten_k, flatten_v = self.flatten_kv_cache(
                k_cache,
                v_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=kv_flatten_size if use_fa3 else out_size,
                out_dtype=query.dtype,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                flatten_kv_layout='shd' if use_fa3 else 'hsd',
            )
            if use_fa3:
                q_rope = query[:, :, self.v_head_size:]
                q_nope = query[:, :, :self.v_head_size]
                k_rope = flatten_k.view(kv_flatten_size, self.num_kv_heads, -1)[:, :, self.v_head_size:]
                c_kv = flatten_k.view(kv_flatten_size, self.num_kv_heads, -1)[:, :, :self.v_head_size]
                from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func
                attn_output = flash_attn_varlen_func(
                    q=q_rope,
                    k=k_rope,
                    v=c_kv,
                    qv=q_nope,
                    cu_seqlens_q=attn_metadata.cu_seqlens_q,
                    cu_seqlens_k=attn_metadata.cu_seqlens_k,
                    max_seqlen_q=max_q_seqlen,
                    max_seqlen_k=kv_flatten_size,
                    softmax_scale=self.scale,
                    causal=self.causal,
                    window_size=(-1, -1) if self.sliding_window is None else self.sliding_window,
                    softcap=-1.0 if self.logit_softcapping is None else self.logit_softcapping,
                )
            else:
                attn_output = query.new_empty(o_shape)
                self.flash_attention_fwd(
                    query,
                    flatten_k,
                    flatten_v,
                    attn_output,
                    q_start_loc=q_start_loc,
                    q_seqlens=q_seqlens,
                    kv_start_loc=kv_start_loc,
                    kv_seqlens=kv_seqlens,
                    max_seqlen=max_q_seqlen,
                    window_size=self.sliding_window,
                    sm_scale=self.scale,
                    logit_softcapping=self.logit_softcapping,
                    causal=self.causal,
                )
        return attn_output


class FA3Impl(TritonAttentionImpl):
    """Triton attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = None,
        causal: bool = True,
        **kwargs,
    ):
        assert alibi is False, 'alibi not supported for FA3'
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            causal=causal,
            **kwargs,
        )
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func
        self.flash_attn_varlen_func_v3 = flash_attn_varlen_func

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        """forward."""
        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        fill_q_start_loc = q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        fill_seqlens = q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        if attn_metadata.is_decoding:
            max_q_seqlen = 1
        else:
            max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        fill_max_q_seqlen = max_q_seqlen
        if attn_metadata.fill_seqlens is not None:
            fill_seqlens = attn_metadata.fill_seqlens
            fill_max_q_seqlen = key.numel() // (key.size(-1) * key.size(-2))
            fill_q_start_loc = fill_seqlens.cumsum(0) - fill_seqlens
        is_decoding = attn_metadata.is_decoding
        # fill kv cache
        if key is not None and value is not None:
            self.fill_kv_cache(
                key,
                value,
                k_cache,
                v_cache,
                fill_q_start_loc,
                fill_seqlens,
                kv_seq_length=kv_seqlens,
                max_q_seq_length=fill_max_q_seqlen,
                block_offsets=block_offsets,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
            )

        q_shape = query.shape
        o_shape = q_shape[:-1] + (self.v_head_size, )
        attn_output = query.new_empty(o_shape)

        if is_decoding:
            self.paged_attention_fwd(
                query,
                k_cache,
                v_cache,
                attn_output,
                block_offsets,
                kv_seqlens=kv_seqlens,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                window_size=self.sliding_window,
                sm_scale=self.scale,
                logit_softcapping=self.logit_softcapping,
            )
        else:
            flatten_k, flatten_v = self.flatten_kv_cache(
                k_cache,
                v_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=kv_flatten_size,
                out_dtype=query.dtype,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                flatten_kv_layout='shd',
            )
            sliding_window = (-1, -1) if self.sliding_window is None else self.sliding_window
            if isinstance(sliding_window, int):
                sliding_window = (sliding_window, sliding_window)
            attn_output = self.flash_attn_varlen_func_v3(
                q=query,
                k=flatten_k,
                v=flatten_v,
                cu_seqlens_q=attn_metadata.cu_seqlens_q,
                cu_seqlens_k=attn_metadata.cu_seqlens_k,
                max_seqlen_q=max_q_seqlen,
                max_seqlen_k=kv_flatten_size,
                softmax_scale=self.scale,
                causal=self.causal,
                window_size=sliding_window,
                softcap=-1.0 if self.logit_softcapping is None else self.logit_softcapping,
            )
        return attn_output


@functools.lru_cache
def _enable_fa3(alibi: bool, learnable_sink: bool, block_sparse_size: int):
    enable = not alibi and not learnable_sink and block_sparse_size == 1
    if enable and not use_fa3_warning():
        enable = False
    return enable


class TritonAttentionBuilder(AttentionBuilder[TritonAttentionMetadata]):
    """Triton attention builder."""

    @staticmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logical_softcapping: float = None,
        causal: bool = True,
        use_flash_mla: bool = False,
        learnable_sink: bool = False,
        block_sparse_size: int = 1,
        **kwargs,
    ) -> TritonAttentionImpl:
        """build."""
        enable_fa3 = _enable_fa3(alibi, learnable_sink, block_sparse_size)
        if use_flash_mla is True:
            logger.debug('Build FlashMLAImpl Attention')
            return FlashMLAImpl(num_heads,
                                head_size,
                                scale=scale,
                                num_kv_heads=num_kv_heads,
                                v_head_size=v_head_size,
                                alibi=alibi,
                                sliding_window=sliding_window,
                                logical_softcapping=logical_softcapping,
                                causal=causal,
                                **kwargs)
        elif enable_fa3:
            logger.debug('Build FA3Impl Attention')
            return FA3Impl(num_heads,
                           head_size,
                           scale=scale,
                           num_kv_heads=num_kv_heads,
                           v_head_size=v_head_size,
                           alibi=alibi,
                           sliding_window=sliding_window,
                           logical_softcapping=logical_softcapping,
                           causal=causal,
                           **kwargs)
        else:
            logger.debug('Build TritonAttentionImpl Attention')
            return TritonAttentionImpl(num_heads,
                                       head_size,
                                       scale=scale,
                                       num_kv_heads=num_kv_heads,
                                       v_head_size=v_head_size,
                                       alibi=alibi,
                                       sliding_window=sliding_window,
                                       logical_softcapping=logical_softcapping,
                                       causal=causal,
                                       block_sparse_size=block_sparse_size,
                                       **kwargs)
