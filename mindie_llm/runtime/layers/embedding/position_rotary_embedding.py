# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
from enum import Enum

import torch

from mindie_llm.runtime.layers.custom_layer import CustomLayer


class PositionEmbeddingType(int, Enum):
    ROPE = 0


class PositionRotaryEmbedding(CustomLayer):
    '''
    Positional Rotary Embedding (RoPE) module with scaling support.
    '''

    def __init__(self, inv_freq: torch.Tensor, scaling_factor: float = 1.0, base: float = 10000.0):
        """
        Initialize rotary embedding with inverse frequencies.

        Args:
            inv_freq: The inverse frequencies for RoPE.
            scaling_factor: Scaling factor for extended context
            base: Base value for frequency calculation (default: 10000).
        """

        super().__init__()

        self.base = base
        self.inv_freq = inv_freq
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_cached_total = None
        self._sin_cached_total = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.scaling_factor = scaling_factor
        self._ntk_alpha_cached = 1.0
        self.position_ids_offset = []
        self.pos_lens = None
        self.ntk_inv_freqs = None
        self.position_ids_expanded = None
        self._batch_seq_len_cached = []
        self._batch_ntk_alpha_cached = []
        self._global_seq_len_cached = {}
        self._global_ntk_alpha_cached = {}
        self._table_size = 0

    @classmethod
    def static(cls, dim: int, base: float, device: torch.device, scaling_factor: float = 1.0):
        """
        Factory method to create an embedding with computed inverse frequencies.

        Args:
            dim: Dimension of the rotary embedding.
            base: Base value for frequency calculation.
            device: The device to create tensors on.
            scaling_factor: Scaling factor for positions.

        Returns:
            Initialized PositionRotaryEmbedding instance
        """

        inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.double) / dim)
        ).to(torch.float)
        return cls(inv_freq, scaling_factor, base)

    def update_cos_sin_cache_total(self, dtype: torch.dtype, device: torch.device, seqlen: int) -> None:
        """Update full-dim RoPE with concatenated frequency doubling (standard variant)."""
        self._update_cos_sin_cache_total(dtype, device, seqlen, duplication_mode="concat")

    def get_cos_cached_total(self) -> torch.Tensor:
        """Return cached full-dimension cosine table."""
        return self._cos_cached_total

    def get_sin_cached_total(self) -> torch.Tensor:
        """Return cached full-dimension sine table."""
        return self._sin_cached_total

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _update_cos_sin_cache_total(
            self, dtype: torch.dtype, device: torch.device,
            seqlen: int, duplication_mode: str = "concat") -> None:
        """
        Update full-dimension cosine and sine caches for RoPE.

        Args:
            dtype: Target dtype (e.g., torch.float16).
            device: Target device (e.g., 'npu:0').
            seqlen: Required sequence length.
            duplication_mode (str): Frequency duplication strategy:
                - "concat": emb = torch.cat((freqs, freqs), dim=-1) → [f0,f1,...,f0,f1,...]
                - "repeat": emb = torch.repeat_interleave(freqs, 2, dim=-1) → [f0,f0,f1,f1,...]
        """

        # Use the same cache variables (both functions write to _cos/sin_cached_total)
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached_total.device != device
                or self._cos_cached_total.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)

            # Note: Cohere version skips the scaling_factor == 0 check;
            # we keep it for safety unless explicitly disabled.
            if hasattr(self, 'scaling_factor') and math.isclose(self.scaling_factor, 0.0):
                raise ValueError(
                    "PositionRotaryEmbedding scaling_factor cannot be 0. "
                    "It is recommended to be >= 1.0. See README.md for rope config guidance."
                )
            t = t / self.scaling_factor
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))

            if duplication_mode == "concat":
                emb = torch.cat((freqs, freqs), dim=-1)
            elif duplication_mode == "repeat":
                emb = torch.repeat_interleave(freqs, 2, dim=-1)
            else:
                raise ValueError(f"Unsupported duplication_mode: {duplication_mode}")

            self._cos_cached_total = torch.cos(emb).to(dtype)
            self._sin_cached_total = torch.sin(emb).to(dtype)