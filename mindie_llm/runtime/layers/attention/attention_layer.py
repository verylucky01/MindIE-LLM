# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import ABC, abstractmethod

import torch

from mindie_llm.runtime.layers.custom_layer import CustomLayer

from .backend import get_attn_backend
from .backend.abstract import AttentionBackend
from . import update_global_attn_dict


class AttentionLayerBase(CustomLayer, ABC):

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        pass


class Attention(AttentionLayerBase):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        attn_backend: type[AttentionBackend] | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.key_cache = None
        self.value_cache = None
        update_global_attn_dict(prefix, self)
        self.prefix = prefix
    
        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                use_mla=False
            )
        else:
            self.attn_backend = attn_backend

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            **extra_impl_args,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        self_kv_cache = (self.key_cache, self.value_cache)
        return self.impl.forward(
            self, query, key, value, self_kv_cache
        )