# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any

import torch

from mindie_llm.runtime.utils.npu.device_utils import DeviceType, get_npu_node_info
from mindie_llm.runtime.layers.fused_moe.token_dispatcher import (
    TokenDispatcherWithAllGather, TokenDispatcherWithMC2, TokenDispatcherWithAll2AllV, MoETokenDispatcher,
    MoeAllGatherArgs, MoeMC2Args, MoeAll2AllArgs)
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.utils.log.logging import logger


_MOE_COMM_DISPATCHER_MAP: Dict[Optional[MoECommType], MoETokenDispatcher] = {}


class MoECommType(Enum):
    ALLGATHER = "allgather"
    MC2 = "mc2"
    ALLTOALL = "all2all"
    FUSED_ALLTOALL = "fused_all2all"


def select_moe_comm_method(quant_type: Optional[str] = None
                          ) -> Optional[MoECommType]:
    """1. If expert parallel is not enabled, we use all-gather since MC2 and all-to-all
    are designed for expert parallelism.
    2. If expert parallel is enabled, we need to consider the soc version and the
    number of tokens. This is based on the observation that all-gather is more
    efficient than all-to-all when running on _910B.

        a. For _910B, we choose from MC2 and all-gather.

        b. For _910_93, we choose from MC2 and all-to-all.

        In both cases, we use MC2 when in prefill phase.

    Args:
        quant_type (Optional[str], optional): The quantization type. Defaults to None.

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType: The selected MoE communication method.
    """
    ascend_device_type = get_npu_node_info().get_device_type()
    parallel_info_manager = get_parallel_info_manager()
    is_prefill = get_forward_context().is_prefill
    enable_expert_parallel = parallel_info_manager.get(ParallelType.MOE_EP).is_enabled()

    # NOTE: Due to unsolved error for mc2 in prefill phase, we disable mc2 in prefill phase.
    # NOTE: Temporarily not use parameter num_tokens to select MoECommType.
    if not enable_expert_parallel:
        moe_comm_type = MoECommType.ALLGATHER
    elif ascend_device_type in {DeviceType.ASCEND_910B}:
        if not is_prefill and parallel_info_manager.world_size >= 16:
            moe_comm_type = MoECommType.MC2
        else:
            if quant_type == "w4a8_dynamic":
                moe_comm_type = MoECommType.ALLTOALL
            else:
                moe_comm_type = MoECommType.ALLGATHER
    elif ascend_device_type in {DeviceType.ASCEND_910_93}:
        moe_comm_type = MoECommType.MC2 if not is_prefill else MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {ascend_device_type}")

    # (note) MoECommType.ALLGATHER: Currently does not support enabling DP.
    if moe_comm_type == MoECommType.ALLGATHER and parallel_info_manager.get(ParallelType.ATTN_DP).is_enabled():
        moe_comm_type = MoECommType.ALLTOALL
        logger.debug('Currently do not support MoECommType.ALLGATHER with dp.')

    # (note) MoECommType.MC2 and MoECommType.ALLTOALL: Do not support moe_tp > 1,
    # and require flash_comm to be enabled.
    if moe_comm_type in (MoECommType.MC2, MoECommType.ALLTOALL):
        if parallel_info_manager.get(ParallelType.MOE_TP).is_enabled():
            err_msg = 'MoECommType.MC2 and MoECommType.ALLTOALL: Do not support moe_tp > 1,'
            # try to change to MoECommType.ALLGATHER
            if parallel_info_manager.get(ParallelType.ATTN_DP).is_enabled():
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            else:
                logger.warning(err_msg)
                moe_comm_type = MoECommType.ALLGATHER

    return moe_comm_type


def get_cached_dispatcher(
        moe_comm_type: Optional[MoECommType]) -> Optional[MoETokenDispatcher]:
    return _MOE_COMM_DISPATCHER_MAP.get(moe_comm_type, None)


# NOTE: setup_moe_comm_method() is a temporary initialization helper. It will be removed in the future.
# _MOE_COMM_DISPATCHER_MAP is intentionally kept as a module-level globals()
# registry that maps MoECommType to its corresponding
# MoETokenDispatcher implementation.
def setup_moe_comm_method():
    _MOE_COMM_DISPATCHER_MAP[MoECommType.ALLTOALL] = TokenDispatcherWithAll2AllV()
    _MOE_COMM_DISPATCHER_MAP[MoECommType.ALLGATHER] = TokenDispatcherWithAllGather()
    _MOE_COMM_DISPATCHER_MAP[MoECommType.MC2] = TokenDispatcherWithMC2()


def build_moe_comm_args(
        *,
        moe_comm_type: MoECommType,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        num_experts: int,
        expert_list: list,
        expert_map: torch.Tensor,
        with_quant: bool,
        mc2_mask: torch.Tensor,
        shared_experts: Any,
        quantized_x_for_share: Any,
        dynamic_scale_for_share: Any
):
    """
        Build MoE dispatch args according to moe_comm_type.
        This function is the ONLY place that knows:
            - which args type belongs to which dispatcher
    """
    if moe_comm_type == MoECommType.ALLGATHER:
        return MoeAllGatherArgs(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            num_experts=num_experts,
            expert_list=expert_list,
            expert_map=expert_map,
            with_quant=with_quant,
        )

    elif moe_comm_type == MoECommType.MC2:
        return MoeMC2Args(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=num_experts,
            with_quant=with_quant,
            mc2_mask=mc2_mask,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
        )

    elif moe_comm_type == MoECommType.ALLTOALL:
        return MoeAll2AllArgs(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=num_experts,
        )

    else:
        raise NotImplementedError(f"moe_comm_type {moe_comm_type} not implemented.")