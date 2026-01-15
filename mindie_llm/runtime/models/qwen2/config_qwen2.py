# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig


@dataclass
class Qwen2Config(HuggingFaceConfig):
    """Configuration class for Qwen2 model.

    Extends HuggingFaceConfig with Qwen2-specific attributes.
    """
    attention_bias = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = self.hidden_size // self.num_attention_heads