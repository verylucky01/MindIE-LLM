# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import patch
from dataclasses import dataclass
import torch
import mindspore

from mindie_llm.text_generator.utils.kvcache_settings import (
    NPUSocInfo,
    calc_block_mem,
    calc_npu_mem,
    gb,
    watch_npu_mem,
    KVCacheSettings,
)
from mindie_llm.modeling.backend_type import BackendType


# 模拟ModelInfo类
@dataclass
class MockModelInfo:
    num_kv_heads: int
    head_size: int
    k_head_size: int = 0
    v_head_size: int = 0
    num_layers: int = 1
    data_byte_size: int = 2
    kvcache_quant_layers: list = None
    dtype: type = torch.float16
    enable_nz: bool = False

    def __post_init__(self):
        if self.kvcache_quant_layers is None:
            self.kvcache_quant_layers = []


class TestNPUSocInfo(unittest.TestCase):
    @patch("acl.get_soc_name")
    def test_support_nz_positive(self, mock_get_soc):
        # 测试支持NZ的SOC名称
        for name in ["910PremiumA", "910ProA", "910A", "910ProB", "910B", "310P"]:
            mock_get_soc.return_value = name
            soc_info = NPUSocInfo()
            self.assertTrue(soc_info.support_nz())

    @patch("acl.get_soc_name")
    def test_support_nz_negative(self, mock_get_soc):
        # 测试不支持NZ的SOC名称
        for name in ["910", "310", "other"]:
            mock_get_soc.return_value = name
            soc_info = NPUSocInfo()
            self.assertFalse(soc_info.support_nz())

    @patch("acl.get_soc_name")
    def test_soc_name_none(self, mock_get_soc):
        mock_get_soc.return_value = None
        soc_info = NPUSocInfo()
        assert soc_info.support_nz() is False


class TestMemoryCalculation(unittest.TestCase):
    def get_base_model_info(self):
        return MockModelInfo(
            num_kv_heads=4,
            head_size=64,
            num_layers=2,
            kvcache_quant_layers=[False, False]
        )

    def test_calc_block_mem_basic(self):
        # 基础计算测试
        base_model_info = self.get_base_model_info()
        block_size = 16
        mem = calc_block_mem(base_model_info, block_size)

        expected = 2 * (4 * 64 + 4 * 64) * 2 * 16
        self.assertEqual(mem, expected)

    def test_calc_block_mem_with_quant(self):
        base_model_info = self.get_base_model_info()
        base_model_info.kvcache_quant_layers = [True, False]
        block_size = 16
        mem = calc_block_mem(base_model_info, block_size)

        layer1 = (4 * 64) * 1 + (4 * 64) * 2
        layer2 = (4 * 64) * 2 + (4 * 64) * 2
        expected = (layer1 + layer2) * 16
        self.assertEqual(mem, expected)

    def test_calc_block_mem_speculative(self):
        base_model_info = self.get_base_model_info()
        mem = calc_block_mem(base_model_info, 16, num_speculative_tokens=1)

        expected = 3 * (4 * 64 + 4 * 64) * 2 * 16
        self.assertEqual(mem, expected)

    def test_calc_npu_mem(self):
        base_model_info = self.get_base_model_info()
        block_nums = 10
        block_size = 16
        npu_mem = calc_npu_mem(block_nums, base_model_info, block_size)
        block_mem = calc_block_mem(base_model_info, block_size)
        self.assertEqual(npu_mem, block_nums * block_mem)

    def test_gb_conversion(self):
        # 测试GB转换
        self.assertEqual(gb(1024 ** 3), 1.0)  # 1GB
        self.assertEqual(gb(2 * 1024 ** 3), 2.0)  # 2GB
        self.assertAlmostEqual(gb(512 * 1024 ** 2), 0.5, places=2)  # 512MB


class TestKVCacheSettings(unittest.TestCase):
    def get_base_model_info(self):
        return MockModelInfo(
            num_kv_heads=4,
            head_size=64,
            num_layers=2,
            kvcache_quant_layers=[False, False]
        )

    @patch("acl.get_soc_name")
    def test_init_basic(self, mock_get_soc):
        mock_get_soc.return_value = "other"  # 不支持NZ
        base_model_info = self.get_base_model_info()
        settings = KVCacheSettings(
            rank=0,
            model_info=base_model_info,
            cpu_mem=1024 ** 3,
            npu_mem=2 * 1024 ** 3,
            block_size=16,
            backend_type=BackendType.ATB,
            is_separated_pd=False
        )

        # 验证基础属性
        self.assertEqual(settings.num_layers, 2)
        self.assertEqual(settings.num_heads, 4)
        self.assertFalse(settings.need_nz)

    @patch("acl.get_soc_name")
    def test_cal_kv_total_head_size(self, mock_get_soc):
        mock_get_soc.return_value = "910A"  # 支持NZ
        base_model_info = self.get_base_model_info()
        base_model_info.enable_nz = True
        settings = KVCacheSettings(
            rank=0,
            model_info=base_model_info,
            cpu_mem=1024 ** 3,
            npu_mem=2 * 1024 ** 3,
            block_size=16,
            backend_type=BackendType.ATB,
            is_separated_pd=False
        )

        total, k_total, v_total, k_quant_total = settings._cal_kv_total_head_size()
        self.assertEqual(total, 256)
        self.assertEqual(k_total, 256)
        self.assertEqual(v_total, 256)
        self.assertEqual(k_quant_total, 256)

    @patch("acl.get_soc_name")
    def test_cal_set_kv_block_shapes(self, mock_get_soc):
        mock_get_soc.return_value = "910A"
        base_model_info = self.get_base_model_info()
        base_model_info.enable_nz = True
        settings = KVCacheSettings(
            rank=0,
            model_info=base_model_info,
            cpu_mem=1024 ** 3,
            npu_mem=2 * 1024 ** 3,
            block_size=16,
            backend_type=BackendType.ATB,
            is_separated_pd=False
        )
        settings._cal_set_kv_block_shapes()

        self.assertEqual(settings.block_shape, (16, 16, 16))
        self.assertEqual(settings.k_block_shape, (16, 16, 16))

    def test_dtype_to_str(self):
        # 测试数据类型转换
        self.assertEqual(
            KVCacheSettings.dtype_to_str(BackendType.ATB, torch.float16),
            "float16"
        )
        self.assertEqual(
            KVCacheSettings.dtype_to_str(BackendType.MS, mindspore.float32),
            "float"
        )
        # 测试不匹配的后端和数据类型
        with self.assertRaises(Exception):
            KVCacheSettings.dtype_to_str(BackendType.ATB, mindspore.float16)

    @patch("acl.rt.get_mem_info")
    @patch("mindie_llm.text_generator.utils.kvcache_settings.npu.synchronize")
    def test_watch_npu_mem(self, mock_sync, mock_get_mem):
        mock_get_mem.return_value = (512 * 1024 ** 2, 1024 ** 3, 0)
        total, peak = watch_npu_mem(0, False, 65536, "success")
        self.assertEqual(total, 1024 ** 3)
        self.assertEqual(peak, 512 * 1024 ** 2)


if __name__ == "__main__":
    unittest.main()