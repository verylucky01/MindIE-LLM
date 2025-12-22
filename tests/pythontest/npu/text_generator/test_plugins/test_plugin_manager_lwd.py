# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import json
import sys
from pathlib import Path
import queue
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from mindie_llm.text_generator.adapter import get_generator_backend, parse_config, ParseType
from mindie_llm.utils.env import ENV
from mindie_llm.modeling.backend_type import BackendType
from mindie_llm.text_generator.generator import Generator
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.config import ModelConfig
from mindie_llm.text_generator.utils.input_metadata import InputMetadata, LwdMetadata
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams
from mindie_llm.text_generator.plugins.plugin_manager_lwd import PluginManagerLwd
from mindie_llm.text_generator.utils.model_input import ModelInputWrapper


class TestPluginLwd(unittest.TestCase):
    def setUp(self):
        generator_backend = Mock()
        kvcache_settings = Mock()
        infer_context = Mock()
        output_filter = Mock()
        is_mix_model = Mock()
        plugin_list = Mock()
        model_role = Mock()
        self.plugin_lwd = PluginManagerLwd(generator_backend, kvcache_settings, infer_context, output_filter, is_mix_model,
            plugin_list, model_role)
        self.plugin_lwd.return_queue = queue.Queue()
        self.plugin_lwd.async_inference = True
        self.plugin_lwd.forward_loop = Mock()
        self.metadata_ = Mock()

    @classmethod
    def setUpClass(cls):
        print("TestPluginLwd start")

    @classmethod
    def tearDownClass(cls):
        print("TestPluginLwd end")

    def test_lwd_sampling_output(self):
        self.metadata_.batch_size = 1
        self.metadata_.all_sequence_ids = Mock()
        self.plugin_lwd.lwd_sampling_output(self.metadata_)

    def test_should_skip_return_tokens(self):
        self.metadata_.is_prefill = True
        self.metadata_.layerwise_disaggregated_exe_stage = Mock()
        self.metadata_.layerwise_disaggregated_exe_stage.start_exec_layer = 0
        self.metadata_.layerwise_disaggregated_exe_stage.end_exec_layer = 0
        self.plugin_lwd.should_skip_return_tokens(self.metadata_)

        self.metadata_.is_prefill = False
        self.metadata_.layerwise_disaggregated_exe_stage = Mock()
        self.metadata_.layerwise_disaggregated_exe_stage.start_exec_layer = 1
        self.metadata_.layerwise_disaggregated_exe_stage.end_exec_layer = 1
        self.plugin_lwd.should_skip_return_tokens(self.metadata_)

    def test_set_clean_sequence_ids(self):
        generator_backend = Mock()
        kvcache_settings = Mock()
        infer_context = Mock()
        output_filter = Mock()
        is_mix_model = Mock()
        plugin_list = Mock()
        model_role = Mock()
        self.plugin_lwd = PluginManagerLwd(generator_backend, kvcache_settings, infer_context, output_filter, is_mix_model,
            plugin_list, model_role)
        infer_context.clear_cache = Mock()
        self.plugin_lwd.set_clean_sequence_ids([100, 101])

    def test_generate_token(self):
        self.plugin_lwd.preprocess = Mock()
        self.plugin_lwd.preprocess.return_value = [Mock(), Mock(), Mock(), Mock()]
        self.plugin_lwd.is_mix_model = True
        self.plugin_lwd.model_inputs_update_manager = Mock()
        self.plugin_lwd.model_inputs_update_manager.return_value = [Mock(), Mock(), Mock()]
        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.sample_preprocess_manager = Mock()
        self.plugin_lwd.postprocess = Mock()

        tmp_metadata = InputMetadata(batch_size=1, is_prefill=True, batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])], batch_max_output_lens=[100, 101], block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101], all_sequence_ids=[100, 101], layerwise_disaggregated_exe_stage=LwdMetadata())
        self.plugin_lwd.generate_token(tmp_metadata)

    def test_generate_token_async_edge(self):
        self.plugin_lwd.preprocess = Mock()
        self.plugin_lwd.preprocess.return_value = [Mock(), Mock(), Mock(), Mock()]
        self.plugin_lwd.is_mix_model = True
        self.plugin_lwd.model_inputs_update_manager = Mock()
        self.plugin_lwd.model_inputs_update_manager.return_value = [Mock(), Mock(), Mock()]
        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.sample_preprocess_manager = Mock()
        self.plugin_lwd.postprocess = Mock()
        self.plugin_lwd.plugin_verify_manager = Mock()

        def read_lock_():
            import threading
            lock_ = threading.Lock()
            return lock_

        def prepare_model_inputs_(model_input, q_lens, attn_mask):
            return [0, 1], {"a": 1}
        generator_backend_ = Mock()
        generator_backend_.get_new_stream = read_lock_
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        generator_backend_.dp = 0
        generator_backend_.forward_from_model_inputs = Mock()
        self.plugin_lwd.generator_backend = generator_backend_

        tmp_metadata = InputMetadata(batch_size=1, is_prefill=True, batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])], batch_max_output_lens=[100, 101], block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101], all_sequence_ids=[100, 101], layerwise_disaggregated_exe_stage=LwdMetadata())
        self.plugin_lwd.generate_token_async_edge(tmp_metadata)

        self.plugin_lwd.is_mix_model = None
        self.plugin_lwd.generate_token_async_edge(tmp_metadata)

    @patch("mindie_llm.text_generator.utils.generation_output.GenerationOutput.fill_dummy")
    def test_generate_token_async_edge1(self, dummy_mock):
        self.plugin_lwd.preprocess = Mock()
        self.plugin_lwd.preprocess.return_value = [Mock(), Mock(), Mock(), Mock()]
        self.plugin_lwd.is_mix_model = True
        self.plugin_lwd.model_inputs_update_manager = Mock()
        self.plugin_lwd.model_inputs_update_manager.return_value = [Mock(), Mock(), Mock()]
        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.sample_preprocess_manager = Mock()
        self.plugin_lwd.postprocess = Mock()
        self.plugin_lwd.plugin_verify_manager = Mock()

        def read_lock_():
            import threading
            lock_ = threading.Lock()
            return lock_

        def prepare_model_inputs_(model_input, q_lens, attn_mask):
            return [0, 1], {"a": 1}
        generator_backend_ = Mock()
        generator_backend_.get_new_stream = read_lock_
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        generator_backend_.dp = 0
        generator_backend_.forward_from_model_inputs = Mock()
        self.plugin_lwd.generator_backend = generator_backend_
        tmp_metadata = InputMetadata(batch_size=1, is_prefill=False, batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])], batch_max_output_lens=[100, 101], block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101], all_sequence_ids=[100, 101], layerwise_disaggregated_exe_stage=LwdMetadata(end_exec_layer=0))
        output_res = Mock()
        output_res.input_metadata = Mock()
        output_res.input_metadata.is_prefill = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage = Mock()
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_exec_layer = 0
        output_res.input_metadata.all_sequence_ids = [0]
        output_res.cache_ids = [0]
        self.plugin_lwd.output_queue.put(output_res)

        self.plugin_lwd.postprocess.return_value = MagicMock()
        self.plugin_lwd.postprocess.return_value.sequence_ids = [0, 1]
        self.plugin_lwd._fill_in_model_result = Mock()
        self.plugin_lwd.clean_sequence_ids = [0]
        res_out = MagicMock()
        res_out.sequence_ids = [0]
        res_out.finish_reason = [0]
        self.plugin_lwd.return_queue.put(res_out)
        self.plugin_lwd.generate_token_async_edge(tmp_metadata)
        self.plugin_lwd.plugin_list.append("mtp")
        self.plugin_lwd.is_mix_model = False
        
        def prepare_model_inputs_(model_input, q_lens, spec_mask, sub_model_inputs, hidden_states):
            return [0, 1], {"a": 1}
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        self.plugin_lwd.generate_token_async_edge(tmp_metadata)

    @patch("mindie_llm.text_generator.utils.generation_output.GenerationOutput.fill_dummy")
    def test_generate_token_async_cloud(self, dummy_mock):
        self.plugin_lwd.preprocess = Mock()
        self.plugin_lwd.preprocess.return_value = [Mock(), Mock(), Mock(), Mock()]
        self.plugin_lwd.is_mix_model = True
        self.plugin_lwd.model_inputs_update_manager = Mock()
        self.plugin_lwd.model_inputs_update_manager.return_value = [Mock(), Mock(), Mock()]
        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.sample_preprocess_manager = Mock()
        self.plugin_lwd.postprocess = Mock()
        self.plugin_lwd.plugin_verify_manager = Mock()

        def read_lock_():
            import threading
            lock_ = threading.Lock()
            return lock_

        def prepare_model_inputs_(model_input, q_lens, attn_mask):
            return [0, 1], {"a": 1}
        generator_backend_ = Mock()
        generator_backend_.get_new_stream = read_lock_
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        generator_backend_.dp = 0
        generator_backend_.forward_from_model_inputs = Mock()
        self.plugin_lwd.generator_backend = generator_backend_
        tmp_metadata = InputMetadata(batch_size=1, is_prefill=True, batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])], batch_max_output_lens=[100, 101], block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101], all_sequence_ids=[100, 101], layerwise_disaggregated_exe_stage=LwdMetadata())
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)

        self.plugin_lwd.is_mix_model = None
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)

        output_res = Mock()
        output_res.input_metadata = Mock()
        output_res.input_metadata.layerwise_disaggregated_exe_stage = Mock()
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_exec_layer = 0
        output_res.cache_ids = [0]
        self.plugin_lwd.output_queue.put(output_res)
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)

        self.plugin_lwd.role_type = "slave"
        tmp_val1 = MagicMock()
        tmp_val1.cpu = MagicMock()
        tmp_val = MagicMock()
        tmp_val.logits = [[tmp_val1]]
        self.plugin_lwd.generator_backend.forward_from_model_inputs = MagicMock(return_value=tmp_val)
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)
        
        self.plugin_lwd.plugin_list.append("mtp")
        
        def prepare_model_inputs_(model_input, q_lens, spec_mask, sub_model_inputs, hidden_states):
            return [0, 1], {"a": 1}
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        self.plugin_lwd.output_queue.put(output_res)
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)
        
    def test_prepare_inputs_for_longseq_chunk(self):
        layerwise_disaggregated_exe_stage = LwdMetadata()
        input_metadata = Mock()
        input_metadata.layerwise_disaggregated_exe_stage = layerwise_disaggregated_exe_stage
        model_input_wrapper = Mock(spec=ModelInputWrapper)
        model_input_wrapper.model_kwargs = {}
        model_input_wrapper.input_metadata = input_metadata
        model_input_wrapper.model_inputs = Mock()
        model_input_wrapper.model_inputs.input_lengths = [None] * 128
        model_input_wrapper.model_inputs.lm_head_indices = [None] * 128
        model_input_wrapper.model_inputs.input_ids = [None] * 128
        model_input_wrapper.model_inputs.position_ids = [None] * 128
        model_input_wrapper.model_inputs.block_tables = np.zeros((1,128), dtype=np.int32)
        model_input_wrapper.model_inputs.slots = [None] * 128
        self.plugin_lwd.generator_backend.block_size = 2
        self.plugin_lwd.prepare_inputs_for_longseq_chunk(model_input_wrapper)
        model_input_wrapper.input_metadata.layerwise_disaggregated_exe_stage.long_seq_start_idx = 1
        self.plugin_lwd.prepare_inputs_for_longseq_chunk(model_input_wrapper)

if __name__ == "__main__":
    unittest.main()