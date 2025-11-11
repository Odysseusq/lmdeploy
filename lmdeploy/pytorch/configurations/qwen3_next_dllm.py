# Copyright (c) OpenMMLab. All rights reserved.
from .default import AutoModelConfigBuilder, DefaultModelConfigBuilder


class Qwen3NextModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'qwen3_next'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        cfg.dllm_mask_token = 151700
        cfg.model_paradigm = 'dllm'
        return cfg
