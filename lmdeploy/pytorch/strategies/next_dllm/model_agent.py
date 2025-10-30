# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
from torch.profiler import record_function

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.model_inputs import ModelInputs

from ..base.model_agent import ExtraInputs, ExtraOutputs, ModelAgentStrategy, StoppingCriteria
from .unmasking import UnmaskingProcessor

SeqList = List[SchedulerSequence]


@dataclass
class NextDLLMExtraInputs(ExtraInputs):
    """NextDLLM extra inputs."""
    next_dllm_mask: torch.Tensor

    def broadcast(self, src: int, group, async_op=False):
        return dist.broadcast(self.next_dllm_mask, src=src, group=group, async_op=async_op)


@dataclass
class NextDLLMExtraOutputs(ExtraOutputs):
    """Ar extra outputs."""
    next_dllm_mask: torch.Tensor


def _check_stopwords_next_dllm(token_ids: torch.Tensor, stop_words: torch.Tensor, is_unmasked: torch.Tensor,
                          stopped: torch.Tensor, stop_pos: torch.Tensor, num_appendable_ids: torch.Tensor,
                          output_start_pos: torch.Tensor, inputs: ModelInputs):
    num_tokens = token_ids.size(0)
    batch_size = num_appendable_ids.size(0)
    block_size = num_tokens // batch_size

    # blocks might contain stop words in prev-round chat
    # these stop words should be ignored
    kv_seqlens = inputs.history_lengths + inputs.seq_length
    ignore_pos = (output_start_pos - (kv_seqlens - block_size)).clamp_min(0)
    ignore_range = torch.arange(0, block_size, dtype=ignore_pos.dtype, device=ignore_pos.device)
    ignore_mask = (ignore_range[None, :] < ignore_pos[:, None]).flatten()
    token_ids = token_ids.clone()
    token_ids[ignore_mask] = -1

    # find stop words
    sw_stopped = (token_ids[:, None] == stop_words).any(1)
    sw_stopped = sw_stopped.view(batch_size, block_size)
    sw_stop_pos = sw_stopped.int().argmax(1)

    stop_pos = torch.where(stopped, stop_pos, sw_stop_pos)
    sw_stopped = sw_stopped.any(dim=1)
    sw_stopped = sw_stopped & is_unmasked
    stopped = stopped | sw_stopped

    # update num_appendable_ids
    one_ids = torch.clamp_max(num_appendable_ids, 0)
    num_appendable_ids = torch.where(sw_stopped, one_ids, num_appendable_ids)

    return stopped, stop_pos, num_appendable_ids


@dataclass
class NextDLLMStoppingCriteria(StoppingCriteria):
    num_appendable_ids: torch.Tensor
    output_start_pos: torch.Tensor

    @record_function('stopping_criteria')
    def step(self,
             token_ids: torch.Tensor,
             stop_words: torch.Tensor,
             inputs: Optional[ModelInputs] = None,
             extra_inputs: Optional[NextDLLMExtraInputs] = None):
        """Check whether to stop generation."""
        num_appendable_ids = self.num_appendable_ids
        output_start_pos = self.output_start_pos
        num_tokens = token_ids.size(0)
        batch_size = num_appendable_ids.size(0)
        block_size = num_tokens // batch_size

        next_dllm_mask = extra_inputs.next_dllm_mask
        next_dllm_mask = next_dllm_mask.view(batch_size, block_size)
        is_unmasked = (next_dllm_mask == consts.DLLM_UNMASKED).all(dim=1)

        # check stop by num_new_tokens
        num_appendable_ids -= is_unmasked * block_size
        stopped = num_appendable_ids <= 0
        stop_pos = block_size - 1 + num_appendable_ids

        # check stop words
        if stop_words is not None:
            stopped, stop_pos, num_appendable_ids = _check_stopwords_next_dllm(token_ids,
                                                                          stop_words,
                                                                          is_unmasked,
                                                                          stopped,
                                                                          stop_pos,
                                                                          num_appendable_ids,
                                                                          output_start_pos=output_start_pos,
                                                                          inputs=inputs)

        new_stopping = NextDLLMStoppingCriteria(num_appendable_ids=num_appendable_ids, output_start_pos=output_start_pos)
        return stopped, stop_pos, new_stopping


class NextDLLMModelAgentStrategy(ModelAgentStrategy):

    def __init__(self, next_dllm_config: DLLMConfig, next_dllm_mask_token: int):
        block_size = next_dllm_config.block_length
        self.block_size = block_size
        self.next_dllm_mask_token = next_dllm_mask_token

        self.unmasking_processor = UnmaskingProcessor(next_dllm_config=next_dllm_config)

    def _update_next_dllm(self, next_token_ids: torch.Tensor, next_dllm_mask: torch.Tensor, seqlens: torch.Tensor):
        """Update token_ids and next_dllm_mask."""
        next_dllm_mask_token = self.next_dllm_mask_token
        next_dllm_block_length = self.block_size

        # reshape to (batch, next_dllm_block_length)
        next_token_ids = next_token_ids.view(-1, next_dllm_block_length).clone()
        next_dllm_mask = next_dllm_mask.view(-1, next_dllm_block_length).clone()

        # flags
        is_cached = (next_dllm_mask == consts.DLLM_CACHED).all(dim=1)

        is_masked = (next_dllm_mask == consts.DLLM_MASKED)
        next_token_ids[is_cached[:, None] | is_masked] = next_dllm_mask_token
        next_dllm_mask[is_cached] = consts.DLLM_MASKED
        seqlens = torch.where(is_cached.view(-1), seqlens, seqlens.new_zeros((1, )))

        return next_token_ids.flatten(), next_dllm_mask.flatten(), seqlens

    def slice_outputs(self, inputs: torch.Tensor, seq_length: torch.LongTensor) -> torch.Tensor:
        """Slice outputs."""
        block_length = self.block_size
        # batch size = 1
        if len(seq_length) == 1:
            return inputs[-block_length:]

        if len(seq_length) * block_length == inputs.size(0):
            return inputs
        last_idx = seq_length.cumsum(0)
        block_range = torch.arange(-block_length, 0, device=last_idx.device)
        index = (last_idx[:, None] + block_range[None, :]).flatten()
        inputs = inputs[index]
        return inputs

    def slice_extra_inputs(self, extra_inputs: NextDLLMExtraInputs, seq_length: torch.LongTensor) -> NextDLLMExtraInputs:
        """Slice outputs."""
        next_dllm_mask = self.slice_outputs(extra_inputs.next_dllm_mask, seq_length)
        return NextDLLMExtraInputs(next_dllm_mask=next_dllm_mask)

    def _step_sampling_inputs(self, sampling_inputs: SamplingInputs, next_token_ids: torch.Tensor,
                              next_dllm_mask: torch.Tensor, **kwargs):
        """step."""
        from lmdeploy.pytorch import consts
        next_dllm_block_size = self.block_size
        DLLM_UNMASKED = consts.DLLM_UNMASKED
        is_unmasked = (next_dllm_mask == DLLM_UNMASKED).view(-1, next_dllm_block_size).all(dim=1, keepdim=True)
        num_ignore_eos = sampling_inputs.num_ignore_eos.view(-1, next_dllm_block_size)
        num_ignore_eos = torch.where(is_unmasked, num_ignore_eos - next_dllm_block_size, num_ignore_eos)
        sampling_inputs.num_ignore_eos = num_ignore_eos.flatten()
        return sampling_inputs

    def make_stopping_criteria(self, seqs: SeqList) -> NextDLLMStoppingCriteria:
        """Create stopping criteria."""
        # num_appendable
        num_appendable = [seq.sampling_param.max_new_tokens - seq.num_new_tokens for seq in seqs]
        num_appendable = torch.tensor(num_appendable)
        block_size = self.block_size
        remain = [seq.num_valid_ids % block_size for seq in seqs]
        num_appendable += torch.tensor(remain)

        # output_start_pos
        pos = [seq.output_start_pos for seq in seqs]
        output_start_pos = torch.tensor(pos)

        return NextDLLMStoppingCriteria(num_appendable_ids=num_appendable, output_start_pos=output_start_pos)

    def make_extra_inputs(self, seqs: 'SeqList') -> ExtraInputs:
        """Create extra inputs."""
        next_dllm_masks = [seq.next_dllm_mask for seq in seqs]
        next_dllm_masks = torch.as_tensor(np.concatenate(next_dllm_masks))
        return NextDLLMExtraInputs(next_dllm_mask=next_dllm_masks)

    def make_extra_outputs(self, extra_inputs: NextDLLMExtraInputs) -> NextDLLMExtraOutputs:
        """Create extra outputs."""
        next_dllm_mask = extra_inputs.next_dllm_mask
        return NextDLLMExtraOutputs(next_dllm_mask=next_dllm_mask)

    def update_inputs_for_next_step(self, model_inputs: 'ModelInputs', sampling_inputs: 'SamplingInputs',
                                    next_token_ids: torch.Tensor, model_metas: Any, extra_inputs: NextDLLMExtraInputs,
                                    **kwargs):
        """Step next inputs."""
        model_inputs.model_metas = model_metas
        next_dllm_mask = extra_inputs.next_dllm_mask

        next_token_ids, next_dllm_mask, step_seqlens = self._update_next_dllm(next_token_ids, next_dllm_mask, model_inputs.seq_length)
        model_inputs.step(next_token_ids, step_seqlens)
        self._step_sampling_inputs(sampling_inputs, next_token_ids, next_dllm_mask=next_dllm_mask)

        extra_inputs = NextDLLMExtraInputs(next_dllm_mask=next_dllm_mask)
        return model_inputs, extra_inputs

    def post_sampling(self, inputs: 'ModelInputs', logits: torch.Tensor, next_token_ids: torch.LongTensor,
                      extra_inputs: NextDLLMExtraInputs):
        """Post sampling."""
        next_dllm_mask = extra_inputs.next_dllm_mask
        input_ids = inputs.input_ids
        input_ids = self.slice_outputs(input_ids.flatten(), inputs.seq_length)

        next_dllm_mask, next_token_ids = self.unmasking_processor(logits, input_ids, next_token_ids, next_dllm_mask)

        extra_inputs.next_dllm_mask = next_dllm_mask
        return next_token_ids, extra_inputs

    def make_dummy_next_token(self, inputs: 'ModelInputs', logits: torch.Tensor, extra_inputs: NextDLLMExtraInputs):
        """Make dummy next token for broadcast."""
        with torch.inference_mode():
            next_token_ids = inputs.input_ids.new_zeros(logits.size(0))
        return next_token_ids, extra_inputs

    @contextmanager
    def broadcast_next_token(self, next_token_ids: torch.Tensor, extra_inputs: NextDLLMExtraInputs, dist_ctx: DistContext):
        """Broadcast next token ids and extra inputs."""
        tp_gpu_group = dist_ctx.tp_gpu_group
        dist.broadcast(next_token_ids, src=0, group=tp_gpu_group, async_op=True)
        handle = extra_inputs.broadcast(src=0, group=tp_gpu_group, async_op=True)
        yield
        handle.wait()

