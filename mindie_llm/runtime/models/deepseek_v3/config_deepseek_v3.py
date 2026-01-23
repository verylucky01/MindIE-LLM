# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional, Any
from dataclasses import dataclass

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig
from mindie_llm.runtime.utils.helpers.parameter_validators import IntParameterValidator, Field
from mindie_llm.utils.log.error_code import ErrorCode
from mindie_llm.utils.log.logging import logger


@dataclass
class DeepseekV3Config(HuggingFaceConfig):
    model_type: str = "deepseekv3"
    vocab_size: int = 102400
    hidden_size: int = 5120
    intermediate_size: int = 12288
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 60
    num_attention_heads: int = 128
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 163840
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 100000
    eos_token_id: int = 100001
    rope_theta: float = 10000.0
    start_reasoning_token_id: int = 128798
    end_reasoning_token_id: int = 128799
    is_nzcasted: bool = False

    def __init__(self, rope_scaling, **kwargs):
        # (NOTE): delete kwargs in the future
        super().__init__(**kwargs)
        self.attribute_map = {
            'head_dim': 'qk_nope_head_dim',
        }
        # (NOTE): add default value for compatibility
        if "ep_level" not in kwargs:
            self.ep_level = 1
        if self.model_type in ["deepseek_v3"]:
            self.is_reasoning_model = True
        # (NOTE): add validation
        self.rope_scaling_dict = rope_scaling
        # (NOTE): get from config
        self.index_n_heads = 64
        self.index_head_dim = 128
        self.index_topk = 2048

    def validate(self):
        super().validate()
        
        validators = {
            'num_experts_per_tok': IntParameterValidator(Field(ge=1, le=256), allow_none=True),
            'n_shared_experts': IntParameterValidator(Field(ge=0, le=256), allow_none=True),
            'first_k_dense_replace': IntParameterValidator(Field(ge=0, le=61), allow_none=False),
            'n_routed_experts': IntParameterValidator(Field(ge=2, le=256), allow_none=True),
            'q_lora_rank': IntParameterValidator(Field(ge=1, le=1536), allow_none=True),
            'qk_nope_head_dim': IntParameterValidator(Field(ge=1, le=128), allow_none=True),
            'qk_rope_head_dim': IntParameterValidator(Field(ge=1, le=64), allow_none=True)
        }

        for key, validator in validators.items():
            value = getattr(self, key)
            validator.validate(value, key)         

        if getattr(self, "num_experts_per_tok") > getattr(self, "n_routed_experts"):
            msg = f"self.num_experts_per_tok should be less than self.n_routed_experts, " \
                  f"but {self.num_experts_per_tok=}, {self.n_routed_experts=}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if getattr(self, "first_k_dense_replace") > getattr(self, "num_hidden_layers"):
            msg = f"self.first_k_dense_replace should be less than self.num_hidden_layers, " \
                  f"but {self.first_k_dense_replace=}, {self.num_hidden_layers=}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if self.topk_method not in ["greedy", "group_limited_greedy", "noaux_tc"]:
            msg = "`topk_method`'s type field must be one of ['greedy', 'group_limited_greedy', 'noaux_tc'], " \
                  f"got {self.topk_method}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        # Validate top-k parameter consistency
        if self.topk_method == "greedy" and self.topk_group != self.n_group and self.n_group != 1:
            msg = f"`topk_method is `greedy`, please set `topk_group=1` and `n_group=1`, " \
                  f"got topk_group={self.topk_group}, n_group={self.n_group}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
