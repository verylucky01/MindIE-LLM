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

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig, RopeScaling


class TestHuggingFaceConfig(unittest.TestCase):

    def test_from_dict_creates_instance(self):
        config = HuggingFaceConfig.from_dict({
            "max_position_embeddings": 2048,
            "vocab_size": 32000,
        })
        self.assertIsInstance(config, HuggingFaceConfig)
        self.assertEqual(config.max_position_embeddings, 2048)
        self.assertEqual(config.vocab_size, 32000)

    def test_valid_rope_scaling_passes_validation(self):
        config = HuggingFaceConfig.from_dict({
            "max_position_embeddings": 2048,
            "vocab_size": 32000,
            "rope_scaling": {
                "factor": 4.0,
                "rope_type": "yarn",
                "original_max_position_embeddings": 4096,
                "beta_fast": 64,
                "beta_slow": 2
            }
        })
        # Expect no error
        config.validate()

    def test_invalid_rope_type_raises(self):
        with self.assertRaises(ValueError) as cm:
            HuggingFaceConfig.from_dict({
                "num_attention_heads": 32,
                "rms_norm_eps": 0.1,
                "max_position_embeddings": 2048,
                "vocab_size": 32000,
                "rope_scaling": {"rope_type": "invalid_type"}
            })

    def test_invalid_factor_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            HuggingFaceConfig.from_dict({
                "max_position_embeddings": 2048,
                "vocab_size": 32000,
                "rope_scaling": {"factor": 1e6}
            })

    def test_invalid_max_position_embeddings_negative_raises(self):
        with self.assertRaises(ValueError):
            HuggingFaceConfig.from_dict({
                "num_attention_heads": 32,
                "rms_norm_eps": 0.1,
                "vocab_size": 32000,
                "max_position_embeddings": -10
            })

    def test_invalid_vocab_size_zero_raises(self):
        with self.assertRaises(ValueError):
            HuggingFaceConfig.from_dict({
                "num_attention_heads": 32,
                "rms_norm_eps": 0.1,
                "vocab_size": 0,
                "max_position_embeddings": 2048
            })

    def test_none_rope_scaling_allowed(self):
        config = HuggingFaceConfig.from_dict({
            "vocab_size": 2800,
            "max_position_embeddings": 2048,
            "rope_scaling": None
        })
        self.assertIsInstance(config.rope_scaling, RopeScaling)


if __name__ == '__main__':
    unittest.main()