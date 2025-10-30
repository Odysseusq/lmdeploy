# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from lmdeploy.pytorch.config import DLLMConfig, ModelConfig
from lmdeploy.pytorch.strategies.base.sequence import SequenceStrategy
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.pytorch.strategies.base.cudagraph import CudagraphStrategy
    from lmdeploy.pytorch.strategies.base.model_inputs import ModelInputsStrategy
    from lmdeploy.pytorch.strategies.base.sampling import SamplingStrategy
    from lmdeploy.pytorch.strategies.base.model_agent import ModelAgentStrategy
    from lmdeploy.pytorch.strategies.base.engine import EngineStrategy
    from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

from ..base import StrategyFactoryBase

logger = get_logger('lmdeploy')


class NextDLLMStrategyFactory(StrategyFactoryBase):

    def __init__(self, model_config: ModelConfig, next_dllm_config: DLLMConfig):
        """config."""
        self.model_config = model_config
        self.next_dllm_config = next_dllm_config

        # update next_dllm_block_length
        self.next_dllm_block_length = self._update_next_dllm_block_length()

    def _update_next_dllm_block_length(self):
        """Update next_dllm_block_length."""
        if self.next_dllm_config.block_length is None:
            next_dllm_block_length = self.model_config.dllm_block_length
            if next_dllm_block_length is None:
                next_dllm_block_length = 4
                logger.warning('Model does not provide next_dllm_block_length. '
                               f'Set next_dllm_block_length={next_dllm_block_length} as default.')
        else:
            next_dllm_block_length = self.next_dllm_config.block_length

        assert next_dllm_block_length is not None, 'next_dllm_block_length should be set in model_config or next_dllm_config'

        self.next_dllm_config.block_length = next_dllm_block_length
        self.model_config.dllm_block_length = next_dllm_block_length

        if self.next_dllm_config.denoising_steps is None:
            self.next_dllm_config.denoising_steps = next_dllm_block_length
        return next_dllm_block_length

    def build_cudagraph_strategy(self) -> 'CudagraphStrategy':
        """Build cudagraph strategy."""
        from .cudagraph import NextDLLMCudagraphStrategy
        return NextDLLMCudagraphStrategy(block_size=self.next_dllm_block_length)

    def build_sampling_strategy(self) -> 'SamplingStrategy':
        """Build sampling strategy."""
        from .sampling import NextDLLMSamplingStrategy
        pad_token_id = self.model_config.bos_token_id
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        return NextDLLMSamplingStrategy(pad_token_id, self.next_dllm_block_length)

    def build_model_inputs_strategy(self) -> 'ModelInputsStrategy':
        """Build model inputs strategy."""
        from .model_inputs import NextDLLMModelInputsStrategy
        return NextDLLMModelInputsStrategy(block_size=self.next_dllm_block_length)

    def build_model_agent_strategy(self) -> 'ModelAgentStrategy':
        """Build model agent strategy."""
        from .model_agent import NextDLLMModelAgentStrategy
        return NextDLLMModelAgentStrategy(next_dllm_config=self.next_dllm_config, next_dllm_mask_token=self.model_config.dllm_mask_token)

    def build_engine_strategy(self, cache_config: 'CacheConfig',
                              scheduler_config: 'SchedulerConfig') -> 'EngineStrategy':
        """Build engine strategy."""
        from .engine import NextDLLMEngineStrategy
        return NextDLLMEngineStrategy(cache_config=cache_config,
                                  scheduler_config=scheduler_config,
                                  next_dllm_block_length=self.next_dllm_block_length)

    def build_sequence_strategy(self) -> SequenceStrategy:
        from .sequence import NextDLLMSequenceStrategy
        return NextDLLMSequenceStrategy(block_size=self.next_dllm_block_length,
                                    next_dllm_mask_token=self.model_config.dllm_mask_token)

