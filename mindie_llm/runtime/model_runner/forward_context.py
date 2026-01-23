# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass, fields
import torch


@dataclass
class ModuleMetadata:
    @classmethod
    def from_dict(cls, dict_data):
        field_names = {field.name for field in fields(cls)}
        filtered_dict = {k: v for k, v in dict_data.items() if k in field_names}

        return cls(**filtered_dict)


@dataclass
class AttentionMetadata(ModuleMetadata):
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    attn_mask: torch.Tensor
    cos_table: torch.Tensor
    sin_table: torch.Tensor
    seq_lens_list: list | None = None


@dataclass
class LMHeadMetadata(ModuleMetadata):
    lm_head_indices: torch.Tensor


@dataclass
class MtpMetadata(ModuleMetadata):
    last_hidden_states: torch.Tensor = None


@dataclass
class ForwardContext:
    attn_metadata: dict[str, AttentionMetadata | torch.Tensor]
    lmhead_metadata: LMHeadMetadata
    mtp_metadata: MtpMetadata
    is_prefill: bool
    num_tokens_across_dp_cpu: torch.Tensor
    # following ones for aclgraph
    capturing: bool = False  # default eager mode
    num_tokens: int = 0
    num_actual_tokens: int = 0
    # (NOTE): rope generate in DeepSeekV3Model;
    seq_lens: torch.Tensor = None     
    mc2_mask: torch.Tensor = None


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    if _forward_context is None:
        raise RuntimeError(
            "Forward context is not set. "
            "Please use `set_forward_context` to set the forward context."
        )
    return _forward_context


def set_forward_context(context: ForwardContext):
    """Set the current forward context."""
    global _forward_context
    _forward_context = context


def create_forward_context(
    input_metadata: dict, capturing: bool = False
):
    attn_metadata = AttentionMetadata.from_dict(input_metadata)
    lmhead_metadata = LMHeadMetadata.from_dict(input_metadata)
    mtp_metadata = MtpMetadata.from_dict(input_metadata)
    is_prefill = input_metadata["is_prefill"]
    num_tokens = input_metadata.get("num_tokens", 0)
    num_actual_tokens = input_metadata.get("num_actual_tokens", 0)
    seq_lens = input_metadata["seq_lens"]
    num_tokens_across_dp_cpu = input_metadata["num_tokens_across_dp_cpu"]
    mc2_mask = input_metadata.get("mc2_mask", None)

    return ForwardContext(
        attn_metadata=attn_metadata,
        lmhead_metadata=lmhead_metadata,
        mtp_metadata=mtp_metadata,
        is_prefill=is_prefill,
        num_tokens=num_tokens,
        num_actual_tokens=num_actual_tokens,
        capturing=capturing,
        seq_lens=seq_lens,
        num_tokens_across_dp_cpu=num_tokens_across_dp_cpu,
        mc2_mask=mc2_mask,
    )