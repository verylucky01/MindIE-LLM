#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import patch, MagicMock, Mock
import torch
import numpy as np

from mindie_llm.text_generator.utils.batch_context import DictContext, NdarrayContext, BatchContext
from mindie_llm.text_generator.utils.kvcache_settings import KVCacheSettings
from mindie_llm.text_generator.utils.config import ContextParams, CacheConfig, SpCpParallelInfo, DEFAULT_SAMPLING_PARAMS
from mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess import SplitFusePreprocess


class TestSplitFusePreprocess(unittest.TestCase):
    
    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_init(self):
        infer_context = MagicMock()
        model_wrapper = MagicMock()
        model_wrapper.device = 'npu:0'
        kvcache_settings = None
        
        splitfuse_preprocess = SplitFusePreprocess(infer_context, model_wrapper, kvcache_settings)

        self.assertIsNone(splitfuse_preprocess.model_wrapper.device)
        self.assertIsNone(splitfuse_preprocess.device)
    
    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_make_attention_mask(self):
        infer_context = MagicMock()
        model_wrapper = MagicMock()
        model_wrapper.device = 'npu:0'
        kvcache_settings = None
        splitfuse_preprocess = SplitFusePreprocess(infer_context, model_wrapper, kvcache_settings)
        model_inputs = None
        input_metadata = None
        q_lens = None
        hit_mask = None

        req_mask = splitfuse_preprocess.make_attention_mask(model_inputs, input_metadata, q_lens, hit_mask)

        self.assertIsNone(req_mask)

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='atb')
    def test_make_attention_mask_is300i(self):
        infer_context = MagicMock()
        model_wrapper = MagicMock()
        model_wrapper.device = 'npu:0'
        model_wrapper.model_runner = MagicMock()
        model_wrapper.model_runner.attn_mask = MagicMock()
        model_wrapper.model_runner.attn_mask.get_attn_mask.return_value = torch.ones(5, 5)
        kvcache_settings = MagicMock()
        kvcache_settings.dtype = torch.float16
        splitfuse_preprocess = SplitFusePreprocess(infer_context, model_wrapper, kvcache_settings)
        splitfuse_preprocess.is_300i = True
        splitfuse_preprocess.async_inference = False
        model_inputs = MagicMock()
        model_inputs.max_seq_len = 1
        model_inputs.context_length = [3, 3]
        input_metadata = MagicMock()
        input_metadata.is_prefill = True
        q_lens = [2, 2]
        hit_mask = None

        req_mask = splitfuse_preprocess.make_attention_mask(model_inputs, input_metadata, q_lens, hit_mask)

        golden_mask = torch.ones(4, 5)
        self.assertTrue(torch.allclose(req_mask, golden_mask))

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_get_mix_decode_cache_without_hit_mask(self):
        self.device = "npu"
        self.kvcache_settings = Mock(spec=KVCacheSettings)
        self.kvcache_settings.num_npu_blocks = 2
        self.kvcache_settings.block_size = 4
        self.batch_config = CacheConfig(
            cache_size=4,
            pad_token_id=0,
            max_seq_len=10,
            max_gen_len=10
        )
        self.spcp_info = SpCpParallelInfo(
            sp_parallel_info=Mock(group_size=1, rank=0),
            cp_parallel_info=Mock(group_size=1, rank=0)
        )
        self.context_params = ContextParams(distributed=False)
        self.batch_ctx = BatchContext(
            kvcache_settings=self.kvcache_settings,
            context_params=self.context_params,
            batch_context_config=self.batch_config,
            spcp_parallel_info=self.spcp_info,
            device=self.device
        )
        cache_ids = torch.tensor([0, 2])
        decode_idx = 1
        self.batch_ctx.all_ndarray_context.last_input_ids = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
        self.batch_ctx.all_ndarray_context.seq_lens = torch.tensor([2, 3, 1])
        self.batch_ctx.all_ndarray_context.last_position_ids = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.batch_ctx.all_ndarray_context.used_block_idx = torch.tensor([0, 1, 2])
        self.batch_ctx.all_ndarray_context.used_block_offset = torch.tensor([1, 0, 2])
        self.batch_ctx.kv_slots = torch.arange(500).reshape(50, 10)
        metadata = MagicMock()
        metadata.batch_block_tables = torch.tensor([
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33]
        ])
        result = self.batch_ctx.get_mix_decode_cache_for_splitfuse(cache_ids, decode_idx, metadata, hit_mask=None)
        input_ids, max_seq_len, position_ids, input_lengths, slots = result
        torch.testing.assert_close(input_ids, torch.tensor([[10, 11, 12], [30, 31, 32]]))
        self.assertEqual(max_seq_len, 2)
        torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2], [0, 1, 2]]))
        torch.testing.assert_close(input_lengths, torch.tensor([2, 1]))
        expected_slots = torch.tensor([
            self.batch_ctx.kv_slots[20, 1],
            self.batch_ctx.kv_slots[22, 2]
        ])
        torch.testing.assert_close(slots, expected_slots)

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_get_mix_decode_cache_with_hit_mask(self):
        
        cache_ids = torch.tensor([0, 1])
        decode_idx = 0
        hit_mask = torch.tensor([[True, False]])
        self.device = "npu"
        self.kvcache_settings = Mock(spec=KVCacheSettings)
        self.kvcache_settings.num_npu_blocks = 2
        self.kvcache_settings.block_size = 4
        self.batch_config = CacheConfig(
            cache_size=4,
            pad_token_id=0,
            max_seq_len=10,
            max_gen_len=10
        )
        self.spcp_info = SpCpParallelInfo(
            sp_parallel_info=Mock(group_size=1, rank=0),
            cp_parallel_info=Mock(group_size=1, rank=0)
        )
        self.context_params = ContextParams(distributed=False)
        self.batch_ctx = BatchContext(
            kvcache_settings=self.kvcache_settings,
            context_params=self.context_params,
            batch_context_config=self.batch_config,
            spcp_parallel_info=self.spcp_info,
            device=self.device
        )
        self.batch_ctx.spcp_parallel_info.scp_rank = 0

        self.batch_ctx.all_ndarray_context.last_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.batch_ctx.all_ndarray_context.seq_lens = torch.tensor([3, 2])
        self.batch_ctx.all_ndarray_context.last_position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
        self.batch_ctx.all_ndarray_context.cpu_cached_seq_idx = torch.tensor([[4], [5]])
        self.batch_ctx.batch_context_config.max_block_size = 4
        self.batch_ctx.kv_slots = torch.arange(1000).reshape(100, 10)
        metadata = MagicMock()
        metadata.batch_block_tables = torch.tensor([
            [50, 51, 52, 53],
            [60, 61, 62, 63]
        ])

        result = self.batch_ctx.get_mix_decode_cache_for_splitfuse(cache_ids, decode_idx, metadata, hit_mask=hit_mask)
        input_ids, max_seq_len, position_ids, input_lengths, slots = result
        torch.testing.assert_close(input_ids, torch.tensor([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(max_seq_len, 3)
        torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2], [0, 1, 2]]))
        torch.testing.assert_close(input_lengths, torch.tensor([3, 2]))
        expected_slots = torch.tensor([self.batch_ctx.kv_slots[51, 1], self.batch_ctx.kv_slots[51, 1]])
        torch.testing.assert_close(slots, expected_slots)


if __name__ == '__main__':
    unittest.main()
