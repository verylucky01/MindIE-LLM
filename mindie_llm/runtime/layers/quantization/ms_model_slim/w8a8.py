# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List

import torch
import torch_npu
from torch import nn


from mindie_llm.runtime.layers.linear.linear_method_base import LinearMethodBase
from mindie_llm.runtime.layers.parameter import (
    BiasParameter,
    ModelWeightParameter,
    ScalerParameter,
    PerTensorScaleParameter,
)
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import InferenceMode
from mindie_llm.runtime.utils.npu_utils import get_platform_info
from mindie_llm.utils.log.logging import logger


SUPPORT_NZ_NPU_LIST = ("Ascend910B3", "Ascend910B4_1", "Ascend910_9381", "Ascend910_9372")


class W8A8PerTensorLinearMethod(LinearMethodBase):
    """
    Implements per-tensor weight and activation quantization (W8A8).
    """
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Creates and registers quantized weights and scales.

        Args:
            layer: The layer to register parameters in.
            input_size_per_partition: Input dimension for this partition.
            output_partition_sizes: List of output dimensions for each partition.
            bias: Whether to create bias parameters.
            weight_dtype: Data type for weights.
            bias_dtype: Data type for bias.
            **extra_weight_attrs: Additional attributes for parameters.
        """
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8),
        )
        weight.add_attrs({
            self.INPUT_DIM: 1,
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs,
        })

        input_scale = ScalerParameter(data=torch.empty(1, dtype=weight_dtype))
        input_scale.add_attrs(extra_weight_attrs)

        input_offset = ScalerParameter(data=torch.empty(1, dtype=torch.int8))
        input_offset.add_attrs(extra_weight_attrs)

        deq_scale_dtype = weight_dtype
        if weight_dtype == torch.float16:
            deq_scale_dtype = torch.int64
        elif weight_dtype == torch.bfloat16:
            deq_scale_dtype = torch.float32
        else:
            raise ValueError(f"Dtype {weight_dtype} is not supported in `W8A8PerTensorLinearMethod`.")
        deq_scale = PerTensorScaleParameter(data=torch.empty(sum(output_partition_sizes), dtype=deq_scale_dtype))
        deq_scale.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})

        quant_bias = BiasParameter(data=torch.empty(sum(output_partition_sizes), dtype=torch.int32))
        quant_bias.add_attrs({self.INPUT_DIM: 0, self.OUTPUT_DIM: 0, **extra_weight_attrs})

        layer.register_parameter("weight", weight)
        layer.register_parameter("input_scale", input_scale)
        layer.register_parameter("input_offset", input_offset)
        layer.register_parameter("deq_scale", deq_scale)
        layer.register_parameter("quant_bias", quant_bias)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies per-tensor quantization and matrix multiplication.

        Args:
            layer: The layer containing quantization parameters.
            x: Input tensor.
        Returns:
            Output tensor after quantized matmul.
        """

        # Quantize input tensor to 8-bit signed integer (qint8) using per-tensor non-symmetric quantization.
        # Parameters:
        #   layer.input_scale.data: Scale factor for quantization (per-tensor)
        #   layer.input_offset.data: Zero-point offset for non-symmetric quantization
        #   torch.qint8: Target quantization data type (8-bit signed integer)
        #   -1: Quantization axis (whole tensor, no per-dimension quantization)
        #   False: Non-symmetric quantization flag (uses offset; True would be symmetric)
        input_tensor_quant = torch_npu.npu_quantize(
            x, layer.input_scale.data, layer.input_offset.data, torch.qint8, -1, False)
        out = torch_npu.npu_quant_matmul(
            input_tensor_quant, layer.weight.data, layer.deq_scale.data,
            bias=layer.quant_bias.data, output_dtype=layer.weight_dtype)
        return out

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        expanding_factor = layer.weight.data.shape[1]
        layer.input_scale.data = \
            1 / layer.input_scale.data.repeat(expanding_factor).to(layer.weight_dtype).contiguous().npu()
        layer.input_offset.data = \
            layer.input_offset.data.repeat(expanding_factor).to(layer.weight_dtype).contiguous().npu()
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()

        soc_name = get_platform_info().soc_name
        if soc_name in SUPPORT_NZ_NPU_LIST: 
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
            logger.debug("Convert weight to FRACTAL_NZ done, current format is %s", 
                       torch_npu.get_npu_format(layer.weight.data))


class W8A8PerTokenLinearMethod(LinearMethodBase):
    """
    Implements per-token activation quantization with per-tensor weight quantization (W8A8).
    """
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """
        Creates and registers quantized weights and scales.

        Args:
            layer: The layer to register parameters in.
            input_size_per_partition: Input dimension for this partition.
            output_partition_sizes: List of output dimensions for each partition.
            bias: Whether to create bias parameters.
            weight_dtype: Data type for weights.
            bias_dtype: Data type for bias.
            **extra_weight_attrs: Additional attributes for parameters.
        """
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8),
        )
        weight.add_attrs({self.INPUT_DIM: 1, self.OUTPUT_DIM: 0, **extra_weight_attrs})

        weight_scale_type = torch.float32 if weight_dtype == torch.float16 else torch.bfloat16
        weight_scale = PerTensorScaleParameter(
            data=torch.empty(sum(output_partition_sizes), 1, dtype=weight_scale_type),
        )
        weight_scale.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})

        weight_offset = PerTensorScaleParameter(
            data=torch.empty(sum(output_partition_sizes), 1, dtype=torch.float16),
        )
        weight_offset.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})

        enable_anti_outlier = True
        try:
            layer.quant_config.get_quant_type_by_weight_name(layer.prefix, self.BIAS)
        except ValueError:
            enable_anti_outlier = False
        if enable_anti_outlier:
            bias = BiasParameter(torch.zeros(sum(output_partition_sizes), dtype=bias_dtype))
            bias.add_attrs({"output_dim": 0, **extra_weight_attrs})
            layer.register_parameter(self.BIAS, bias)
        else:
            layer.register_parameter(self.BIAS, None)

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_offset", weight_offset)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_tensor_quant, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        out = torch_npu.npu_quant_matmul(
            input_tensor_quant, layer.weight.data, layer.weight_scale.data,
            pertoken_scale=pertoken_scale, bias=None, output_dtype=layer.weight_dtype)
        if layer.bias is not None:
            out = out + layer.bias.data
        return out

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()

        soc_name = get_platform_info().soc_name
        if soc_name in SUPPORT_NZ_NPU_LIST:
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
            logger.debug("Convert weight to FRACTAL_NZ done, current format is %s", 
                       torch_npu.get_npu_format(layer.weight.data))


class W8A8MixLinearMethod(LinearMethodBase):
    """
    Implements mixed W8A8 quantization using per-tensor for decode and per-token for prefill.
    """
    quant_method = {
        InferenceMode.PREFILL: W8A8PerTokenLinearMethod(),
        InferenceMode.DECODE: W8A8PerTensorLinearMethod(),
    }

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """
        Creates and registers quantized weights for both prefill and decode modes.

        Args:
            layer: The layer to register parameters in.
            input_size_per_partition: Input dimension for this partition.
            output_partition_sizes: List of output dimensions for each partition.
            bias: Whether to create bias parameters.
            weight_dtype: Data type for weights.
            bias_dtype: Data type for bias.
            **extra_weight_attrs: Additional attributes for parameters.
        """
        for quant_method in self.quant_method.values():
            quant_method.create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                bias,
                weight_dtype,
                bias_dtype,
                **extra_weight_attrs,
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: Temprorarily set `is_prefill` to true, wait for `get_forward_context` ready to replace
        is_prefill = True
        if is_prefill:
            result = self.quant_method[InferenceMode.PREFILL].apply(layer, x)
        else:
            result = self.quant_method[InferenceMode.DECODE].apply(layer, x)
        return result

    def process_weights_after_loading(self, layer: nn.Module):
        expanding_factor = layer.weight.data.shape[1]
        layer.input_scale.data = \
            1 / layer.input_scale.data.repeat(expanding_factor).to(layer.weight_dtype).contiguous().npu()
        layer.input_offset.data = \
            layer.input_offset.data.repeat(expanding_factor).to(layer.weight_dtype).contiguous().npu()
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()

        soc_name = get_platform_info().soc_name
        if soc_name in SUPPORT_NZ_NPU_LIST:
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
            logger.debug("Convert weight to FRACTAL_NZ done, current format is %s", 
                       torch_npu.get_npu_format(layer.weight.data))
