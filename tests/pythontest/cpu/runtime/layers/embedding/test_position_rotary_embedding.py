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
import unittest
import torch

from mindie_llm.runtime.layers.embedding.position_rotary_embedding import (
    PositionRotaryEmbedding,
    PositionEmbeddingType,
)


class TestPositionEmbeddingType(unittest.TestCase):
    """Test cases for PositionEmbeddingType enum."""

    def test_enum_values(self):
        """Test enum values."""
        self.assertEqual(PositionEmbeddingType.ROPE, 0)


class TestPositionRotaryEmbedding(unittest.TestCase):
    """Test cases for PositionRotaryEmbedding."""

    def setUp(self):
        """Set up test fixtures."""
        self.dim = 32
        self.base = 10000.0
        self.device = torch.device('cpu')
        self.scaling_factor = 1.0
        # Create inv_freq on CPU
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=self.device, dtype=torch.double) / self.dim)
        ).float()
        self.embedding = PositionRotaryEmbedding(
            self.inv_freq, self.scaling_factor, self.base
        )

    def mock_configuration(self):
        mock_config = type('Config', (), {
                        'rope_theta': 10000.0,
                        'hidden_size': 128,
                        'num_attention_heads': 4,
                        'max_position_embeddings': 2048,
                        'rope_scaling': type('RopeScaling', (), {
                            'factor': 8.0,
                            'original_max_position_embeddings': None,
                            'beta_fast': 32.0,
                            'beta_slow': 1.0
                        })
                    })
        
        seqlen = 100
        device = self.device
        return mock_config, seqlen, device

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.embedding.base, self.base)
        self.assertTrue(torch.allclose(self.embedding.inv_freq, self.inv_freq))
        self.assertEqual(self.embedding.scaling_factor, self.scaling_factor)
        self.assertEqual(self.embedding._seq_len_cached, 0)
        self.assertIsNone(self.embedding._cos_cached)
        self.assertIsNone(self.embedding._sin_cached)
        self.assertIsNone(self.embedding._cos_cached_total)
        self.assertIsNone(self.embedding._sin_cached_total)
        self.assertIsNone(self.embedding._cos_k_cached)
        self.assertIsNone(self.embedding._sin_k_cached)
        self.assertEqual(self.embedding._ntk_alpha_cached, 1.0)
        self.assertEqual(self.embedding.position_ids_offset, [])
        self.assertEqual(self.embedding._table_size, 0)

    def test_init_with_custom_scaling_factor(self):
        """Test initialization with custom scaling factor."""
        scaling_factor = 2.0
        embedding = PositionRotaryEmbedding(
            self.inv_freq, scaling_factor, self.base
        )
        self.assertEqual(embedding.scaling_factor, scaling_factor)

    def test_init_with_custom_base(self):
        """Test initialization with custom base."""
        base = 5000.0
        embedding = PositionRotaryEmbedding(self.inv_freq, self.scaling_factor, base)
        self.assertEqual(embedding.base, base)

    def test_static_method(self):
        """Test static method."""
        embed = PositionRotaryEmbedding.static(
            self.dim, self.base, self.device, self.scaling_factor
        )
        
        self.assertIsInstance(embed, PositionRotaryEmbedding)
        self.assertEqual(embed.base, self.base)
        self.assertEqual(embed.scaling_factor, self.scaling_factor)
        # Verify inv_freq is created on CPU
        self.assertEqual(embed.inv_freq.device, self.device)

    def test_static_method_with_different_scaling_factor(self):
        """Test static method with different scaling factor."""
        scaling_factor = 2.0
        embed = PositionRotaryEmbedding.static(
            self.dim, self.base, self.device, scaling_factor
        )
        self.assertEqual(embed.scaling_factor, scaling_factor)

    def test_static_method_with_different_base(self):
        """Test static method with different base."""
        base = 5000.0
        embed = PositionRotaryEmbedding.static(
            self.dim, base, self.device, self.scaling_factor
        )
        self.assertEqual(embed.base, base)

    def test_update_cos_sin_cache_total_basic(self):
        """Test update_cos_sin_cache_total basic functionality."""
        seqlen = 10
        dtype = torch.float32
        
        self.embedding.update_cos_sin_cache_total(dtype, self.device, seqlen)
        
        self.assertEqual(self.embedding._seq_len_cached, seqlen)
        self.assertIsNotNone(self.embedding._cos_cached_total)
        self.assertIsNotNone(self.embedding._sin_cached_total)
        self.assertEqual(self.embedding._cos_cached_total.shape, (seqlen, self.dim))
        self.assertEqual(self.embedding._sin_cached_total.shape, (seqlen, self.dim))
        # Verify tensors are on CPU
        self.assertEqual(self.embedding._cos_cached_total.device, self.device)
        self.assertEqual(self.embedding._sin_cached_total.device, self.device)

    def test_update_cos_sin_cache_total_scaling_factor_zero(self):
        """Test update_cos_sin_cache_total raises error when scaling_factor is 0."""
        self.embedding.scaling_factor = 0.0
        
        with self.assertRaises(ValueError) as context:
            self.embedding.update_cos_sin_cache_total(torch.float32, self.device, 10)
        
        self.assertIn("scaling_factor cannot be 0", str(context.exception))

    def test_update_cos_sin_cache_total_cache_reuse(self):
        """Test update_cos_sin_cache_total reuses cache when conditions don't change."""
        seqlen = 10
        dtype = torch.float32
        
        self.embedding.update_cos_sin_cache_total(dtype, self.device, seqlen)
        cos_first = self.embedding._cos_cached_total.clone()
        sin_first = self.embedding._sin_cached_total.clone()
        
        # Call again with same seqlen - should reuse cache
        self.embedding.update_cos_sin_cache_total(dtype, self.device, seqlen)
        
        self.assertTrue(torch.allclose(self.embedding._cos_cached_total, cos_first))
        self.assertTrue(torch.allclose(self.embedding._sin_cached_total, sin_first))

    def test_update_cos_sin_cache_total_increase_seqlen(self):
        """Test update_cos_sin_cache_total updates cache when seqlen increases."""
        dtype = torch.float32
        
        self.embedding.update_cos_sin_cache_total(dtype, self.device, 10)
        
        # Increase seqlen - should update cache
        self.embedding.update_cos_sin_cache_total(dtype, self.device, 20)
        
        self.assertEqual(self.embedding._seq_len_cached, 20)
        self.assertEqual(self.embedding._cos_cached_total.shape[0], 20)

    def test_get_cos_cached_total(self):
        """Test get_cos_cached_total."""
        seqlen = 10
        dtype = torch.float32
        
        self.embedding.update_cos_sin_cache_total(dtype, self.device, seqlen)
        
        cos = self.embedding.get_cos_cached_total()
        
        self.assertIsNotNone(cos)
        self.assertTrue(torch.allclose(cos, self.embedding._cos_cached_total))
        self.assertEqual(cos.device, self.device)

    def test_get_sin_cached_total(self):
        """Test get_sin_cached_total."""
        seqlen = 10
        dtype = torch.float32
        
        self.embedding.update_cos_sin_cache_total(dtype, self.device, seqlen)
        
        sin = self.embedding.get_sin_cached_total()
        
        self.assertIsNotNone(sin)
        self.assertTrue(torch.allclose(sin, self.embedding._sin_cached_total))
        self.assertEqual(sin.device, self.device)


    def test_update_cos_sin_cache_total_dtype_change(self):
        """Test update_cos_sin_cache_total updates cache when dtype changes."""
        self.embedding.update_cos_sin_cache_total(torch.float32, self.device, 10)
        
        # Change dtype - should update cache
        self.embedding.update_cos_sin_cache_total(torch.float16, self.device, 10)
        
        self.assertEqual(self.embedding._cos_cached_total.dtype, torch.float16)
        self.assertEqual(self.embedding._sin_cached_total.dtype, torch.float16)

if __name__ == '__main__':
    unittest.main()
