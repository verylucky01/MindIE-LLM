# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

from mindie_llm.runtime.layers.parameter import (
    BaseParameter,
    ModelWeightParameter,
    BiasParameter,
)

from mindie_llm.runtime.layers.quantization.unquantized import (
    UnquantizedLinearMethod, UnquantizedEmbeddingMethod, UnquantizedNormMethod, 
    UnquantizedLayerNormBiasMethod
)


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_config = None
        self.prefix = ""
        self.skip_bias_add = False
        self.enable_anti_outlier = False
        self.variance_epsilon = 1e-6


class MockModule_withBias(MockModule):
    def __init__(self):
        super().__init__()
        self.bias = None


class MockModule_FusedMoed(MockModule):
    def __init__(self):
        super().__init__()
        self.gate_up_weight = None
        self.down_weight = None


class TestUnquantizedLinearMethod(unittest.TestCase):
    
    def setUp(self):
        self.input_size_per_partition = 512
        self.output_partition_sizes = [256, 256]
        self.weight_dtype = torch.float16
        self.bias_dtype = torch.float16
        self.extra_attrs = {"attr1": "value1"}
        self.method = UnquantizedLinearMethod()
        # UnquantizedLinearMethod中create_weight需要创建bias为None或者实际值，apply必然有bias
        self.layer_for_create = MockModule()
        self.layer_for_apply = MockModule_withBias()

    # 无bias且无quant_config场景，bias为none
    def test_create_weights_without_bias(self):
        self.method.create_weights(
            self.layer_for_create,
            self.input_size_per_partition,
            self.output_partition_sizes,
            bias=False,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.bias_dtype,
            **self.extra_attrs
        )

        self.assertTrue(hasattr(self.layer_for_create, 'weight'))
        self.assertEqual(self.layer_for_create.weight.data.shape, (512, 512))  # sum([256, 256]) x 512
        self.assertEqual(self.layer_for_create.weight.data.dtype, self.weight_dtype)
        self.assertEqual(self.layer_for_create.weight.input_dim, 1)
        self.assertEqual(self.layer_for_create.weight.output_dim, 0)
        self.assertEqual(self.layer_for_create.weight.attr1, "value1")

        self.assertTrue(not hasattr(self.layer_for_create, 'bias') or self.layer_for_create.bias is None)

    def test_create_weights_with_bias(self):
        self.method.create_weights(
            self.layer_for_create,
            self.input_size_per_partition,
            self.output_partition_sizes,
            bias=True,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.bias_dtype
        )

        self.assertTrue(hasattr(self.layer_for_create, 'weight'))
        self.assertTrue(hasattr(self.layer_for_create, 'bias'))
        self.assertIsInstance(self.layer_for_create.bias, BiasParameter)
        self.assertEqual(self.layer_for_create.bias.data.shape, (512,))
        self.assertEqual(self.layer_for_create.bias.data.dtype, self.bias_dtype)
        self.assertEqual(self.layer_for_create.bias.output_dim, 0)

    # quant_config is not None
    def test_create_weights_with_anti_outlier_enabled(self):
        # 模拟quant_config启用anti_outlier
        mock_quant_config = MagicMock()
        mock_quant_config.get_quant_type_by_weight_name.return_value = "some_quant_type"
        self.layer_for_create.quant_config = mock_quant_config
        self.layer_for_create.prefix = "test_prefix"

        # 即使bias=False，也会创建bias因为anti_outlier
        self.method.create_weights(
            self.layer_for_create,
            self.input_size_per_partition,
            self.output_partition_sizes,
            bias=False,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.bias_dtype
        )

        self.assertTrue(hasattr(self.layer_for_create, 'bias'))
        self.assertIsInstance(self.layer_for_create.bias, BiasParameter)
        self.assertTrue(self.layer_for_create.enable_anti_outlier)

    def test_create_weights_with_anti_outlier_disabled(self):

        # 模拟quant_config但quant_type获取报错，即不启用anti_outlier
        mock_quant_config = MagicMock()
        mock_quant_config.get_quant_type_by_weight_name.side_effect = ValueError("Not found")
        self.layer_for_create.quant_config = mock_quant_config
        self.layer_for_create.prefix = "test_prefix"

        self.method.create_weights(
            self.layer_for_create,
            self.input_size_per_partition,
            self.output_partition_sizes,
            bias=False,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.bias_dtype
        )

        # 创建bias且设置为None
        self.assertTrue(hasattr(self.layer_for_create, 'bias') and self.layer_for_create.bias is None)
        self.assertFalse(self.layer_for_create.enable_anti_outlier)

    def test_apply_without_bias(self):
        self.layer_for_apply.weight = ModelWeightParameter(torch.eye(10))
        
        x = torch.randn(2, 3, 10)
        result = self.method.apply(self.layer_for_apply, x)
        
        # 输出和输入相同
        torch.testing.assert_close(result, x)

    def test_apply_with_bias(self):
        self.layer_for_apply.weight = ModelWeightParameter(torch.eye(10))
        self.layer_for_apply.bias = BiasParameter(torch.ones(10))
        
        x = torch.randn(2, 3, 10)
        result = self.method.apply(self.layer_for_apply, x)
        
        # 输出是输入加上偏置
        expected = x + 1
        torch.testing.assert_close(result, expected)

    def test_apply_with_skip_bias_add_false(self):
        self.layer_for_apply.weight = ModelWeightParameter(torch.eye(10))
        self.layer_for_apply.bias = BiasParameter(torch.ones(10))
        self.layer_for_apply.skip_bias_add = False
        
        x = torch.randn(2, 3, 10)
        result = self.method.apply(self.layer_for_apply, x)
        
        # 输出是输入加上偏置
        expected = x + 1
        torch.testing.assert_close(result, expected)

    def test_apply_with_skip_bias_add_true_and_no_anti_outlier(self):
        self.layer_for_apply.weight = ModelWeightParameter(torch.eye(10))
        self.layer_for_apply.bias = BiasParameter(torch.ones(10))
        self.layer_for_apply.skip_bias_add = True
        self.layer_for_apply.enable_anti_outlier = False
        
        x = torch.randn(2, 3, 10)
        result = self.method.apply(self.layer_for_apply, x)
        
        # 输出和输入相同
        torch.testing.assert_close(result, x)

    def test_apply_with_skip_bias_add_true_and_anti_outlier_enabled(self):
        self.layer_for_apply.weight = ModelWeightParameter(torch.eye(10))
        self.layer_for_apply.bias = BiasParameter(torch.ones(10))
        self.layer_for_apply.skip_bias_add = True
        self.layer_for_apply.enable_anti_outlier = True
        
        x = torch.randn(2, 3, 10)
        result = self.method.apply(self.layer_for_apply, x)
        
        # 输出是输入加上偏置
        expected = x + 1
        torch.testing.assert_close(result, expected)


class TestUnquantizedEmbeddingMethod(unittest.TestCase):
    
    def setUp(self):
        self.method = UnquantizedEmbeddingMethod()
        self.layer = MockModule()

    def test_create_weights(self):
        input_size_per_partition = 1000
        output_partition_sizes = [512, 512]
        input_size = 1000
        output_size = 1024
        params_dtype = torch.float16
        extra_attrs = {"attr2": "value2"}

        self.method.create_weights(
            self.layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_attrs
        )

        self.assertTrue(hasattr(self.layer, 'weight'))
        self.assertIsInstance(self.layer.weight, ModelWeightParameter)
        self.assertEqual(self.layer.weight.data.shape, (1000, 1024))  # 1000 x sum([512, 512])
        self.assertEqual(self.layer.weight.data.dtype, params_dtype)
        self.assertEqual(self.layer.weight.input_dim, 1)
        self.assertEqual(self.layer.weight.output_dim, 0)
        self.assertEqual(self.layer.weight.attr2, "value2")

    def test_apply(self):
        # 设置权重
        embedding_table = torch.randn(100, 50)
        self.layer.weight = ModelWeightParameter(embedding_table)
        
        # 创建索引张量
        indices = torch.randint(0, 100, (2, 3))
        result = self.method.apply(self.layer, indices)
        
        # 验证结果形状
        self.assertEqual(result.shape, (2, 3, 50))
        
        # 验证结果值是否正确
        expected = torch.nn.functional.embedding(indices, embedding_table)
        torch.testing.assert_close(result, expected)


class TestUnquantizedNormMethod(unittest.TestCase):
    
    def setUp(self):
        self.method = UnquantizedNormMethod()
        self.layer = MockModule()

    def test_create_weights(self):
        hidden_size = 512
        params_dtype = torch.float16
        extra_attrs = {"attr3": "value3"}

        self.method.create_weights(
            self.layer,
            hidden_size,
            params_dtype,
            **extra_attrs
        )

        self.assertTrue(hasattr(self.layer, 'weight'))
        self.assertIsInstance(self.layer.weight, BaseParameter)
        self.assertEqual(self.layer.weight.data.shape, (512,))
        self.assertEqual(self.layer.weight.data.dtype, params_dtype)
        self.assertEqual(self.layer.weight.attr3, "value3")


    def test_apply_without_residual(self):
        self.layer.weight = BaseParameter(torch.ones(10))
        self.layer.variance_epsilon = 1e-6
        x = torch.randn(2, 3, 10)

        with patch('mindie_llm.runtime.layers.quantization.unquantized.torch_npu.npu_rms_norm') \
            as mock_torch_npu_rmsnorm:
            
            mock_return_value = (x,       # 处理后的x
                                 torch.ones(3))  # 中间结果
            mock_torch_npu_rmsnorm.return_value = mock_return_value

            result = self.method.apply(self.layer, x)

            mock_torch_npu_rmsnorm.assert_called_once()
            # 当layer.weight.data为1，variance_epsilon极小时，RMSNorm接近恒等变化，所以直接返回x
            torch.testing.assert_close(result, x)

    def test_apply_with_residual(self):
        self.layer.weight = BaseParameter(torch.ones(10))
        self.layer.variance_epsilon = 1e-6
        x = torch.randn(2, 3, 10)
        residual = torch.randn(2, 3, 10)

        with patch('mindie_llm.runtime.layers.quantization.unquantized.torch_npu.npu_add_rms_norm') \
            as mock_torch_npu_addrmsnorm:
            mock_return_value = (x,       # 处理后的x
                                 torch.ones(5),  # 中间结果
                                 residual)  # 处理后的residual
            mock_torch_npu_addrmsnorm.return_value = mock_return_value
            result = self.method.apply(self.layer, x, residual)

            mock_torch_npu_addrmsnorm.assert_called_once()
            # 当有residual时，应该返回tuple，包含x和residual
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            torch.testing.assert_close(result[0], x)
            torch.testing.assert_close(result[1], residual)


class TestUnquantizedLayerNormBiasMethod(unittest.TestCase):

    def setUp(self):
        self.method = UnquantizedLayerNormBiasMethod()
        self.layer_for_create = MockModule()
        self.layer_for_apply = MockModule_withBias()

    def test_create_weights(self):
        hidden_size = 512
        params_dtype = torch.float16
        extra_attrs = {"attr4": "value4"}

        self.method.create_weights(
            self.layer_for_create,
            hidden_size,
            params_dtype,
            **extra_attrs
        )

        self.assertTrue(hasattr(self.layer_for_create, 'weight'))
        self.assertTrue(hasattr(self.layer_for_create, 'bias'))
        self.assertIsInstance(self.layer_for_create.weight, BaseParameter)
        self.assertIsInstance(self.layer_for_create.bias, BaseParameter)

        self.assertEqual(self.layer_for_create.weight.data.shape, (hidden_size,))
        self.assertEqual(self.layer_for_create.bias.data.shape, (hidden_size,))
        self.assertEqual(self.layer_for_create.weight.data.dtype, params_dtype)
        self.assertEqual(self.layer_for_create.bias.data.dtype, params_dtype)
        self.assertEqual(self.layer_for_create.weight.attr4, "value4")
        self.assertEqual(self.layer_for_create.bias.attr4, "value4")

    def test_apply(self):
        hidden_size = 10
        self.layer_for_apply.weight = BaseParameter(torch.ones(hidden_size))
        self.layer_for_apply.bias = BaseParameter(torch.zeros(hidden_size))
        self.layer_for_apply.variance_epsilon = 1e-6
        x = torch.randn(2, 3, 10)
        dim = 3

        with patch('mindie_llm.runtime.layers.quantization.unquantized.torch.nn.functional.layer_norm') \
            as mock_torch_functional_layernorm:
            self.method.apply(self.layer_for_apply, x, dim)
            mock_torch_functional_layernorm.assert_called_once()

if __name__ == '__main__':
    unittest.main()