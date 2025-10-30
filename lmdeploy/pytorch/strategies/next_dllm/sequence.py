# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from torch import Tensor

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import (HistoryTokenIds, InputEmbeddings, MessageStatus, MultiModalInputs, SamplingParam,
                                       SchedulerSession, UpdateTokenMode, _to_ndarray)

from ..ar.sequence import SchedulerSequenceDefault
from ..base.sequence import SequenceStrategy

SeqList = List['SchedulerSequenceNextDLLM']

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED
DLLM_MASK_DTYPE = np.uint8


class HistoryNextDLLMMask(HistoryTokenIds):

    def __init__(self, token_ids: np.ndarray = None, dtype: np.dtype = DLLM_MASK_DTYPE):
        super().__init__(token_ids=token_ids, dtype=dtype)


@dataclass
class SchedulerSequenceNextDLLM(SchedulerSequenceDefault):

    # For next_dllm
    history_next_dllm_mask: HistoryNextDLLMMask = field(default_factory=HistoryNextDLLMMask)

    def __post_init__(self):
        """Post init."""
        super().__post_init__()
        self._num_valid_ids: int = len(self.history_cache)
        self._strategy: NextDLLMSequenceStrategy = self._seq_meta.strategy

    @property
    def next_dllm_mask(self):
        start = self.num_history_ids
        end = start + self._num_token_ids
        return self.history_next_dllm_mask._token_ids[start:end]

    @property
    def num_valid_ids(self):
        return self._num_valid_ids

    @property
    def generated_ids(self) -> np.ndarray:
        end = self.num_valid_ids
        start = end - self.num_new_tokens
        return self.history_cache._token_ids[start:end]

    @property
    def all_next_dllm_mask(self):
        return self.history_next_dllm_mask._token_ids[:self.num_all_ids]

    @property
    def next_dllm_block_length(self):
        return self._strategy.block_size

    @property
    def next_dllm_mask_token(self):
        return self._strategy.next_dllm_mask_token

    def set_stop_pos(self, pos: int):
        next_dllm_block_length = self.next_dllm_block_length
        val = next_dllm_block_length - pos - 1
        self._num_valid_ids -= val
        self.num_new_tokens -= val

    def _update_token_ids_inputs(self, token_ids: np.ndarray, next_dllm_mask: np.ndarray):
        """Append tokens."""
        num_tokens = len(token_ids)
        next_dllm_block_length = self.next_dllm_block_length
        next_dllm_mask_token = self.next_dllm_mask_token
        new_token_ids = [token_ids]
        new_next_dllm_mask = [next_dllm_mask]

        # add uncached tokens in token_ids
        # for example, [cccc cccc uumm], the [uu] in last block is remain valid.
        num_remain_valid = self.num_valid_ids - self.num_history_ids
        if num_remain_valid != 0:
            prev_token_ids = self.valid_ids[-num_remain_valid:]
            prev_next_dllm_mask = np.full_like(prev_token_ids, DLLM_UNMASKED, dtype=DLLM_MASK_DTYPE)
            new_token_ids = [prev_token_ids] + new_token_ids
            new_next_dllm_mask = [prev_next_dllm_mask] + new_next_dllm_mask
            self.history_cache.resize(self.num_history_ids)
            self.history_next_dllm_mask.resize(self.num_history_ids)
            num_tokens += num_remain_valid

        # pad to align with next_dllm_block_length
        num_pad = (-num_tokens) % next_dllm_block_length
        if num_pad > 0:
            pad_ids = np.full_like(token_ids, next_dllm_mask_token, shape=(num_pad, ))
            pad_mask = np.full_like(next_dllm_mask, DLLM_MASKED, shape=(num_pad, ))
            new_token_ids += [pad_ids]
            new_next_dllm_mask += [pad_mask]

        token_ids = np.concatenate(new_token_ids)
        next_dllm_mask = np.concatenate(new_next_dllm_mask)

        assert len(token_ids) % next_dllm_block_length == 0

        self.history_cache.append(token_ids)
        self.history_next_dllm_mask.append(next_dllm_mask)
        self.output_start_pos = self._num_valid_ids + len(token_ids)
        self._num_valid_ids = self.num_history_ids + num_tokens
        self._num_token_ids = len(token_ids)
        self.num_new_tokens = 0

    def _update_token_ids_decode(self, token_ids: np.ndarray, next_dllm_mask: np.ndarray):
        """Update token ids for decode."""
        num_tokens = len(token_ids)
        next_dllm_block_length = self.next_dllm_block_length
        next_dllm_mask_token = self.next_dllm_mask_token
        assert num_tokens % next_dllm_block_length == 0
        num_history_ids = self.num_history_ids

        token_ids[next_dllm_mask == DLLM_MASKED] = next_dllm_mask_token
        self.history_cache[num_history_ids:] = token_ids
        self.history_next_dllm_mask[num_history_ids:] = next_dllm_mask

        # check if all blocks are cached
        last_mask = next_dllm_mask[-next_dllm_block_length:]
        is_unmasked = np.all(last_mask == DLLM_UNMASKED)
        is_cached = np.all(last_mask == DLLM_CACHED)

        if is_unmasked:
            num_new = next_dllm_block_length - self._num_valid_ids % next_dllm_block_length
            self._num_valid_ids += num_new
            self.num_new_tokens += num_new

        if is_cached:
            # add new block
            new_token_ids = np.full_like(token_ids, next_dllm_mask_token, shape=(next_dllm_block_length, ))
            new_next_dllm_mask = np.full_like(next_dllm_mask, DLLM_MASKED, shape=(next_dllm_block_length, ))
            self.history_cache.append(new_token_ids)
            self.history_next_dllm_mask.append(new_next_dllm_mask)
            self._num_history_ids += self._num_token_ids
            self._num_token_ids = next_dllm_block_length

    def _update_token_ids_prefill(self, token_ids: np.ndarray, next_dllm_mask: np.ndarray):
        """Update token ids for prefill."""
        next_dllm_block_length = self.next_dllm_block_length
        num_history_ids = self.num_history_ids

        # fill input cache
        if self.num_token_ids > next_dllm_block_length:
            end = self.num_token_ids - next_dllm_block_length
            self.history_next_dllm_mask[num_history_ids:end] = DLLM_CACHED
            self._num_history_ids += end
            self._num_token_ids -= end

        # decoding update
        self._update_token_ids_decode(token_ids, next_dllm_mask)

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: List[InputEmbeddings] = None,
                         model_meta: Dict[str, Any] = None,
                         next_dllm_mask: Tensor = None,
                         mode: UpdateTokenMode = UpdateTokenMode.INPUTS,
                         **kwargs):
        """Update token ids, old token ids will be added to history."""
        # update history image nums
        self._update_embeddings(embeddings)

        # update multimodals
        self._update_multimodals(multimodals)

        self.arrive_time = time.perf_counter()

        token_ids: np.ndarray = _to_ndarray(token_ids)
        if next_dllm_mask is None:
            next_dllm_mask = np.full_like(token_ids, DLLM_UNMASKED, dtype=DLLM_MASK_DTYPE)
        next_dllm_mask: np.ndarray = _to_ndarray(next_dllm_mask)

        if mode == UpdateTokenMode.INPUTS:
            self._update_token_ids_inputs(token_ids, next_dllm_mask)
        elif mode == UpdateTokenMode.PREFILL:
            self._update_token_ids_prefill(token_ids, next_dllm_mask)
        else:
            self._update_token_ids_decode(token_ids, next_dllm_mask)

        if model_meta is not None:
            self.model_meta = model_meta


class NextDLLMSequenceStrategy(SequenceStrategy):

    def __init__(self, block_size: int, next_dllm_mask_token: int) -> None:
        self.block_size = block_size
        self.next_dllm_mask_token = next_dllm_mask_token

    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: Optional[MigrationRequest] = None,
                      resp_cache: bool = False,
                      preserve_cache: bool = False) -> 'SchedulerSequenceNextDLLM':
        """Make sequence."""
        return SchedulerSequenceNextDLLM(seq_id=seq_id,
                                     session=session,
                                     sampling_param=sampling_param,
                                     adapter_name=adapter_name,
                                     migration_request=migration_request,
                                     resp_cache=resp_cache,
                                     preserve_cache=preserve_cache)

    def update_running(self, running: SeqList, batched_outputs: BatchedOutputs, is_decoding: bool) -> None:
        """Update running sequences."""
        next_token_ids = batched_outputs.next_token_ids
        stopped = batched_outputs.stopped
        stopped = stopped.tolist()
        model_metas = batched_outputs.model_metas
        if model_metas is None:
            model_metas = [None] * len(running)
        next_dllm_mask = batched_outputs.extra_outputs.next_dllm_mask
        stop_pos = batched_outputs.stop_pos

        batch_size = len(running)
        next_token_ids = next_token_ids.view(batch_size, -1).numpy()
        next_dllm_mask = next_dllm_mask.view(batch_size, -1).numpy()
        stop_pos = stop_pos.tolist()
        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL
        for idx, token in enumerate(next_token_ids):
            msg = running[idx]
            stop = stopped[idx]
            model_meta = model_metas[idx]
            mask = next_dllm_mask[idx]
            if msg.status != MessageStatus.LOCKED:
                continue

            # fill token
            msg.update_token_ids(token, next_dllm_mask=mask, model_meta=model_meta, mode=update_mode)
            if stop:
                msg.set_stop_pos(stop_pos[idx])
                msg.status = MessageStatus.TO_BE_MIGRATED if msg.preserve_cache else MessageStatus.STOPPED

