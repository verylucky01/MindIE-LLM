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
import json
from unittest.mock import MagicMock, patch

import torch

from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.models.qwen2.flash_causal_qwen2 import FlashQwen2ForCausalLM, LwdLayerStatus
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.op_backend import OpBackend
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.adapter_manager import AdapterManager
from atb_llm.models.base.flash_causal_lm import DistributedType
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


LOAD_ATB_SPEED = "atb_llm.models.base.flash_causal_lm.load_atb_speed"
FLASH_QWEN2 = "atb_llm.models.qwen2.flash_causal_qwen2"


class TestFlashQwen2ForCausalLM(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_classes = MockTorchClasses()
        torch.classes = self.torch_classes

        self.config = Qwen2Config(
            model_type="qwen2",
            hidden_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_hidden_layers=28,
            rms_norm_eps=1e-6,
            torch_dtype=torch.float16,
            vocab_size=125696,
            tie_word_embeddings=False,
        )

        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.mapping = Mapping(world_size=2, rank=0)
        self.weights.mapping.attn_tp.rank = 1
        self.weights.process_group = MagicMock()
        self.weights.process_group.rank.return_value = 0
        self.weights.process_group.size.return_value = 2
        self.weights.quant_desc = None

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.TensorHead")
    def test_init(
            self,
            mock_tensor_head,
            mock_load_column_multi,
            mock_qwen_model,
            _mock_init_so
    ) -> None:
        FlashQwen2ForCausalLM(self.config, self.weights)
        mock_qwen_model.assert_called_once_with(
            self.config, self.weights, model_prefix="model", lmhead_prefix="lm_head",
            attn_decode_backend=OpBackend.ATB
        )
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )

        self.config.quantize = "w8a8sc"
        FlashQwen2ForCausalLM(self.config, self.weights)
        mock_tensor_head.load_weight.assert_called_once_with(
            self.config,
            prefix="lm_head",
            weights=self.weights,
            is_norm=False,
        )

        self.config.quantize = ""
        self.config.tie_word_embeddings = True
        FlashQwen2ForCausalLM(self.config, self.weights)
        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=["model.embed_tokens"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.WeightWrapper")
    def test_init_ascend_weight(self, mock_weight_wrapper, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        mock_weight_wrapper_ins = mock_weight_wrapper.return_value
        mock_weight_wrapper_ins.register_embedding = MagicMock()
        mock_weight_wrapper_ins.register_layer = MagicMock()
        mock_weight_wrapper_ins.register_model_norm = MagicMock()
        mock_weight_wrapper_ins.register_model_lmhead = MagicMock()
        mock_weight_wrapper_ins.weights = []
        mock_weight_wrapper_ins.linear_type = {}
        mock_weight_wrapper_ins.pack_quant_type = {}
        mock_weight_wrapper_ins.linear_transpose_types = {}
        mock_weight_wrapper_ins.linear_descs = []

        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.lm_head.linear.trans_flag = True
        ins.graph_manager = MagicMock()
        ins.graph_manager.set_param.return_value = True
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()

        self.config.quantize = QuantType.W8A8_PDMIX
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.graph_manager = MagicMock()
        ins.graph_manager.set_param.return_value = True
        ins.lm_head.linear.trans_flag = True
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()

        ins.adapter_manager = AdapterManager(self.weights)
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_model_prepare_inputs_prefill(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 0], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 3
        golden_kv_cache = [(torch.zeros([1, 128, 1, 128]).npu(), torch.zeros([1, 128, 1, 128]).npu()) for _ in range(2)]
        ins.long_seq_modifier = MagicMock()
        ins.long_seq_modifier.modify_inputs.return_value = True
        ins.qlen_modifier = MagicMock()
        ins.qlen_modifier.modify_inputs.return_value = True
        ins.lora_modifier = MagicMock()
        ins.lora_modifier.modify_inputs.return_value = True
        ins.flash_comm_modifier = MagicMock()
        ins.flash_comm_modifier.modify_inputs.return_value = True
        ins.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(ins.acl_operation_inputs), 12) # 12: base inputs

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_model_prepare_inputs_decode(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 0], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 3
        golden_kv_cache = [(torch.zeros([1, 128, 1, 128]).npu(), torch.zeros([1, 128, 1, 128]).npu()) for _ in range(2)]
        ins.long_seq_modifier = MagicMock()
        ins.long_seq_modifier.modify_inputs.return_value = True
        ins.qlen_modifier = MagicMock()
        ins.qlen_modifier.modify_inputs.return_value = True
        ins.lora_modifier = MagicMock()
        ins.lora_modifier.modify_inputs.return_value = True
        ins.flash_comm_modifier = MagicMock()
        ins.flash_comm_modifier.modify_inputs.return_value = True
        ins.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, False, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(ins.acl_operation_inputs), 12) # 12: base inputs

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_forward(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.graph_manager = MagicMock()
        ins.graph_manager.select_and_execute.return_value = torch.zeros([1024], dtype=torch.float16).npu()
        ins.execute_ascend_operator([], {}, True)
        ins.graph_manager.select_and_execute.assert_called_once()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_init_ascend_kvcache(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.acl_encoder_operation = MagicMock()
        ins.acl_decoder_operation = MagicMock()
        cache = torch.zeros([1, 128, 1, 128], dtype=torch.float16).npu()
        ins.init_kvcache([(cache, cache)])
        self.assertEqual(ins.ascend_kcache_id, id(cache))

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_flash_comm_gate(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.tp_world_size = 2
        ins.soc_info = MagicMock()
        ins.soc_info.soc_version = 255
        ins.soc_info.is_support_hccs.return_value = True
        self.assertTrue(ins.flash_comm_gate(self.weights))

        ins.tp_world_size = 1
        self.assertFalse(ins.flash_comm_gate(self.weights))

        ins.tp_world_size = 2
        ins.soc_info.soc_version = 100
        self.assertFalse(ins.flash_comm_gate(self.weights))

        ins.soc_info.soc_version = 200
        ins.tp_world_size = 8
        self.assertFalse(ins.flash_comm_gate(self.weights))

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_update_matmul_params(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        # A2 + float16 -> aclnn backend
        ins.soc_info.soc_version = 223
        ins.dtype = torch.float16
        ins._update_matmul_params(quantize=QuantType.FLOAT)
        self.assertTrue(ins.aclnn_matmul_backend)
        # W8A8 -> aclnn backend
        ins.aclnn_matmul_backend = False
        ins._update_matmul_params(quantize=QuantType.W8A8)
        self.assertTrue(ins.aclnn_matmul_backend)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_init_kvcache_all_ops(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        # prepare operations
        ins.graph_manager = MagicMock()
        ins.prefix_cache_enable = True
        cache = torch.zeros([1, 128, 1, 128], dtype=torch.float16).npu()
        ins.init_kvcache([(cache, cache)])
        # all available ops should receive kv cache
        ins.graph_manager.set_kv_cache.assert_called_once()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_init_yarn(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        config = Qwen2Config(
            model_type="qwen2",
            hidden_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_hidden_layers=28,
            rms_norm_eps=1e-6,
            torch_dtype=torch.float16,
            vocab_size=125696,
            tie_word_embeddings=False,
        )
        config.rope_scaling = MagicMock()
        config.rope_scaling.type = "yarn"
        config.rope_scaling.factor = None
        with self.assertRaises(ValueError):
            ins = FlashQwen2ForCausalLM(config, self.weights)

        config.rope_scaling.factor = 0.5
        ins = FlashQwen2ForCausalLM(config, self.weights)
        self.assertEqual(ins.mscale, 1)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.TensorHead")
    def test_init_layerwiseDisaggregated(
            self,
            mock_tensor_head,
            mock_load_column_multi,
            mock_qwen_model,
            _mock_init_so
    ) -> None:
        kwargs = {
            "layerwise_disaggregated": "true",
            "layerwise_disaggregated_role_type": "master",
            "inference_mode": MagicMock()
        }
        FlashQwen2ForCausalLM(self.config, self.weights, **kwargs)
        mock_qwen_model.assert_called_with(
            self.config, self.weights, model_prefix="model", lmhead_prefix="lm_head",
            attn_decode_backend=OpBackend.ATB,
            load_list=[0, 27],
            layerwise_disaggregated=True
        )
        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )
        

        kwargs = {
            "layerwise_disaggregated": "true",
            "layerwise_disaggregated_role_type": "slave",
            "inference_mode": MagicMock()
        }
        FlashQwen2ForCausalLM(self.config, self.weights, **kwargs)
        mock_qwen_model.assert_called()
        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )
        
    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.WeightWrapper")
    def test_init_edge_ascend_weight_layerwiseDisaggregated(self, mock_weight_wrapper, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        mock_weight_wrapper_ins = mock_weight_wrapper.return_value
        mock_weight_wrapper_ins.register_embedding = MagicMock()
        mock_weight_wrapper_ins.register_layer = MagicMock()
        mock_weight_wrapper_ins.register_model_norm = MagicMock()
        mock_weight_wrapper_ins.register_model_lmhead = MagicMock()
        mock_weight_wrapper_ins.weights = []
        mock_weight_wrapper_ins.linear_type = {}
        mock_weight_wrapper_ins.pack_quant_type = {}
        mock_weight_wrapper_ins.linear_transpose_types = {}
        mock_weight_wrapper_ins.linear_descs = []
        
        kwargs = {
            "layerwise_disaggregated": "true",
            "layerwise_disaggregated_role_type": "slave",
            "inference_mode": MagicMock()
        }
        ins = FlashQwen2ForCausalLM(self.config, self.weights, **kwargs)
        with patch.object(ins.graph_manager, 'set_param') as mock_set_param:
            ins.init_ascend_weight()

        kwargs = {
            "layerwise_disaggregated": "true",
            "layerwise_disaggregated_role_type": "master",
            "inference_mode": MagicMock()
        }
        ins = FlashQwen2ForCausalLM(self.config, self.weights, **kwargs)
        with patch.object(ins.graph_manager, 'set_param') as mock_set_param:
            ins.init_ascend_weight()

if __name__ == "__main__":
    unittest.main()