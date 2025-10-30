# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.profiler import record_function

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED


class UnmaskingProcessor:

    def __init__(self, next_dllm_config: DLLMConfig):
        self.next_dllm_config = next_dllm_config

    def _get_scores(self, logits: torch.Tensor, token_ids: torch.Tensor):
        """Get scores."""
        scores = logits.softmax(dim=-1)
        scores = scores.gather(-1, token_ids.unsqueeze(-1)).flatten()
        return scores

    def _get_denoise_num(self):
        """Get denoise num."""
        block_size = self.next_dllm_config.block_length
        denoising_steps = self.next_dllm_config.denoising_steps
        if denoising_steps is None:
            denoising_steps = block_size
        num = block_size // self.next_dllm_config.denoising_steps
        num = max(1, min(num, block_size))
        return num

    def low_confidence_static(self, logits: torch.Tensor, token_ids: torch.Tensor, next_dllm_mask: torch.Tensor):
        """static."""
        block_size = self.next_dllm_config.block_length
        topk = self._get_denoise_num()
        scores = self._get_scores(logits, token_ids)
        is_masked = next_dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        next_dllm_mask = next_dllm_mask.view(-1, block_size)
        _, indices = scores.topk(topk, dim=-1)
        next_dllm_unmasked = next_dllm_mask.scatter(-1, indices, DLLM_UNMASKED)

        is_masked = is_masked.view_as(next_dllm_mask)
        next_dllm_mask = torch.where(is_masked, next_dllm_unmasked, next_dllm_mask)
        return next_dllm_mask.flatten()

    def low_confidence_dynamic(self, logits: torch.Tensor, token_ids: torch.Tensor, next_dllm_mask: torch.Tensor):
        """dynamic."""
        block_size = self.next_dllm_config.block_length
        threshold = self.next_dllm_config.confidence_threshold
        scores = self._get_scores(logits, token_ids)
        is_masked = next_dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        next_dllm_mask = next_dllm_mask.view(-1, block_size)
        _, indices = scores.topk(1, dim=-1)
        scores = scores.scatter(-1, indices, threshold)

        is_masked = is_masked.view_as(next_dllm_mask)
        is_masked &= scores >= threshold
        next_dllm_mask[is_masked] = DLLM_UNMASKED
        return next_dllm_mask.flatten()

    def sequential(self, next_dllm_mask: torch.Tensor):
        """sequential."""
        block_size = self.next_dllm_config.block_length
        denoise_num = self._get_denoise_num()
        next_dllm_mask = next_dllm_mask.view(-1, block_size)
        is_masked = next_dllm_mask == DLLM_MASKED

        # get indices
        indices = is_masked.int().argmax(dim=1)
        ranges = torch.arange(0, denoise_num, device=indices.device, dtype=indices.dtype)
        indices = indices[:, None] + ranges[None, :]
        indices = indices % block_size

        next_dllm_unmasked = next_dllm_mask.clone()
        next_dllm_unmasked = next_dllm_unmasked.scatter(-1, indices, DLLM_UNMASKED)
        next_dllm_mask = torch.where(is_masked, next_dllm_unmasked, next_dllm_mask)

        return next_dllm_mask.flatten()

    @record_function('unmasking')
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor, token_ids: torch.Tensor, next_dllm_mask: torch.Tensor):
        """call."""
        strategy = self.next_dllm_config.unmasking_strategy
        if strategy is None:
            return next_dllm_mask

        # reshape to [num_blocks, block_size]
        block_size = self.next_dllm_config.block_length
        next_dllm_mask = next_dllm_mask.unflatten(0, (-1, block_size))

        is_same = (next_dllm_mask == next_dllm_mask[:, :1]).all(dim=1)
        first_mask = next_dllm_mask[:, 0]

        # unmasked to cache
        is_block_unmasked = is_same & (first_mask == DLLM_UNMASKED)
        next_dllm_mask[is_block_unmasked] = DLLM_CACHED

        next_dllm_mask = next_dllm_mask.flatten()
        token_ids = torch.where(next_dllm_mask != DLLM_MASKED, input_ids, token_ids)
        if strategy == UnmaskingStrategy.LOW_CONFIDENCE_STATIC:
            next_dllm_mask = self.low_confidence_static(logits, token_ids, next_dllm_mask)
        elif strategy == UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC:
            next_dllm_mask = self.low_confidence_dynamic(logits, token_ids, next_dllm_mask)
        elif strategy == UnmaskingStrategy.SEQUENTIAL:
            next_dllm_mask = self.sequential(next_dllm_mask)
        else:
            raise RuntimeError(f'strategy {strategy} not supported.')

        return next_dllm_mask, token_ids

