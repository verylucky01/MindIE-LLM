# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import abstractmethod

import torch
from torch import nn

from mindie_llm.runtime.models.base.model_descriptor import ModelDescriptor
from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig


class BaseModelForCausalLM(nn.Module):
    """
    Base causalLM class, which defines the interface that every model must implement
    and the parameters that must be included.

    Attributes:
        model_descriptor (ModelDescriptor): Indicates which features are enabled for this model.
    """
    def __init__(self, mindie_llm_config: MindIELLMConfig):
        super().__init__()

        self.model_descriptor = self._get_model_descriptor_cls().from_config(mindie_llm_config)

    @staticmethod
    def _get_model_descriptor_cls():
        """The default method to get model_descriptor class."""
        return ModelDescriptor

    @abstractmethod    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None
    ):
        """Abstract method to compute hidden_states."""
        pass

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor):
        """Abstract method to compute logits."""
        pass    
