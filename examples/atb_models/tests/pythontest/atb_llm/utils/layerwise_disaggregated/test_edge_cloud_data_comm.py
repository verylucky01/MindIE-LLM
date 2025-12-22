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
import gc
import unittest
from unittest.mock import patch, MagicMock, ANY

import torch
import torch_npu


from atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm import EdgeCloudDataComm
from atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm import SEQ_LEN, BATCH_LEN

MOCKED_INIT_METHOD = f"{__name__}.EdgeCloudDataComm.__init__"


class MockStreamContext:
    def __init__(self, stream):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestEdgeCloudDataComm(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()
        self.original_rank_table = os.environ.get('RANK_TABLE_FILE')
        
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = '2'
        os.environ['RANK'] = '0'
        
        EdgeCloudDataComm._instance = None
        
        self.device_mock = MagicMock()
        torch.npu.set_device = MagicMock(return_value='npu')
        torch_npu.stream = MagicMock(return_value=MagicMock())
        torch_npu.npu.synchronize = MagicMock()
        
        EdgeCloudDataComm.role = "master"
        EdgeCloudDataComm.edge_ip = '127.0.0.1'
        EdgeCloudDataComm.edge_port = "29500"
        EdgeCloudDataComm.rank = 0
        EdgeCloudDataComm.dist_init = False
        EdgeCloudDataComm.dtype = torch.bfloat16

        EdgeCloudDataComm.edge_ranks_num = 8
        EdgeCloudDataComm.cloud_ranks_num = 8
        EdgeCloudDataComm.groups_inter_send_recv = [[0,2],[1,3]]
        EdgeCloudDataComm.map_inter_send_recv = [[0,1],[1,0]]
        
        EdgeCloudDataComm.send_stream = MagicMock()
        EdgeCloudDataComm.recv_stream = MagicMock()
        EdgeCloudDataComm.p_send_card = 0
        EdgeCloudDataComm.d_send_card = 1
        EdgeCloudDataComm.p_recv_card = 0
        EdgeCloudDataComm.d_recv_card = 1
        
        EdgeCloudDataComm.hidden_size = 7168
        
        EdgeCloudDataComm.ret_p = None
        EdgeCloudDataComm.ret_d = None
        EdgeCloudDataComm.need_set_decode_device = False
        EdgeCloudDataComm.need_set_prefill_device = False

        torch.npu.set_device(torch.device(f"npu:{EdgeCloudDataComm.rank % 8}"))

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)
        torch.npu.empty_cache()
        
        EdgeCloudDataComm._instance = None
        gc.collect()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_edge_mode_p_rank_0(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "master"
        mode = 'p'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.groups_inter_send_recv = [[0,2],[1,3]]
        comm.hidden_size = 64
        
        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        peer_index = 1 if comm.role == "master" else 0
        src_rank = comm.p_send_card if mode == 'p' else comm.d_send_card
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            if comm.rank == src_rank:
                dst = comm.map_inter_send_recv[comm.rank][peer_index]
                group_val = comm.groups_inter_send_recv[comm.rank]
                mock_isend.assert_called_once_with(
                    tensor=ANY,
                    dst=dst,
                    group=group_val
                )
            else:
                mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_edge_mode_p_rank_1(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "master"
        mode = 'p'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.groups_inter_send_recv = [[0,2],[1,3]]
        comm.hidden_size = 64

        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_edge_mode_d_rank_0(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "master"
        mode = 'd'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.groups_inter_send_recv = [[0,2],[1,3]]
        comm.hidden_size = 64

        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_edge_mode_d_rank_1(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "master"
        mode = 'd'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1],[1,0]]
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64
        
        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        peer_index = 1 if comm.role == "master" else 0
        src_rank = comm.p_send_card if mode == 'p' else comm.d_send_card
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            if comm.rank == src_rank:
                dst = comm.map_inter_send_recv[comm.rank][peer_index]
                group_val = comm.groups_inter_send_recv[comm.rank]
                mock_isend.assert_called_once_with(
                    tensor=ANY,
                    dst=dst,
                    group=group_val
                )
            else:
                mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_cloud_mode_p_rank_0(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "slave"
        mode = 'p'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.groups_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64
        
        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        peer_index = 1 if comm.role == "master" else 0
        src_rank = comm.p_send_card if mode == 'p' else comm.d_send_card
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            if comm.rank == src_rank:
                dst = comm.map_inter_send_recv[comm.rank][peer_index]
                group_val = comm.groups_inter_send_recv[comm.rank]
                mock_isend.assert_called_once_with(
                    tensor=ANY,
                    dst=dst,
                    group=group_val
                )
            else:
                mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_cloud_mode_p_rank_1(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "slave"
        mode = 'p'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64

        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_cloud_mode_d_rank_0(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "slave"
        mode = 'd'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64

        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_cloud_mode_d_rank_1(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "slave"
        mode = 'd'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1],[1,0]]
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64
        
        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        peer_index = 1 if comm.role == "master" else 0
        src_rank = comm.p_send_card if mode == 'p' else comm.d_send_card
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            if comm.rank == src_rank:
                dst = comm.map_inter_send_recv[comm.rank][peer_index]
                group_val = comm.groups_inter_send_recv[comm.rank]
                mock_isend.assert_called_once_with(
                    tensor=ANY,
                    dst=dst,
                    group=group_val
                )
            else:
                mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_edge_mode_p_rank_0_cat(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "master"
        mode = 'p'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1],[1,0]]
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64

        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        peer_index = 1 if comm.role == "master" else 0
        src_rank = comm.p_send_card if mode == 'p' else comm.d_send_card
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))
        
        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            if comm.rank == src_rank:
                dst = comm.map_inter_send_recv[comm.rank][peer_index]
                group_val = comm.groups_inter_send_recv[comm.rank]
                mock_isend.assert_called_once_with(
                    tensor=ANY,
                    dst=dst,
                    group=group_val
                )
            else:
                mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_send_hidden_cloud_mode_d_rank_1_cat(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "slave"
        mode = 'd'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1],[1,0]]
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64

        test_len = SEQ_LEN if mode == 'p' else BATCH_LEN
        peer_index = 1 if comm.role == "master" else 0
        src_rank = comm.p_send_card if mode == 'p' else comm.d_send_card
        test_tensor = torch.ones((test_len, comm.hidden_size), dtype=torch.bfloat16, device=torch.device("npu"))

        with patch('torch.distributed.isend') as mock_isend:
            comm.send_hidden(mode=mode, out_tensor=test_tensor)
            if comm.rank == src_rank:
                dst = comm.map_inter_send_recv[comm.rank][peer_index]
                group_val = comm.groups_inter_send_recv[comm.rank]
                mock_isend.assert_called_once_with(
                    tensor=ANY,
                    dst=dst,
                    group=group_val
                )
            else:
                mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.warmup_recv')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.warmup_send')
    @patch('torch.distributed.get_rank')
    def test_recv_hidden_edge_mode_p_rank_0(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = "master"
        mode = 'p'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1], [1,0]]
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64
        comm.hccl_comm_warmup(comm.hidden_size)

        peer_index = 1
        src_rank = comm.p_send_card if mode == 'p' else comm.d_send_card

        with patch('torch.distributed.irecv') as mock_isend:
            mock_isend.return_value = 1
            comm.recv_hidden(mode=mode, shape=1024)
            if comm.rank == src_rank:
                src = comm.map_inter_send_recv[comm.rank][peer_index]
                group_val = comm.groups_inter_send_recv[comm.rank]

                mock_isend.assert_called_once_with(
                    ANY,
                    src=src,
                    group=group_val
                )

            else:
                mock_isend.assert_not_called()
            if mode == 'p':
                self.assertEqual(comm.ret_p, 1)
            else:
                self.assertEqual(comm.ret_d, 1)


    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_recv_hidden_edge_mode_p_rank_1(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = 'master'
        mode = 'p'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1], [1,0]]
        comm.hidden_size = 64

        with patch('torch.distributed.irecv', return_value="ret") as mock_irecv:
            comm.recv_hidden(mode=mode, shape=1024)
            mock_irecv.assert_not_called()


    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_recv_hidden_edge_mode_d_rank_0(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = 'master'
        mode = 'd'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1], [1,0]]
        comm.hidden_size = 64

        with patch('torch.distributed.irecv', return_value="ret") as mock_irecv:
            comm.recv_hidden(mode=mode, shape=1024)
            mock_irecv.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.warmup_recv')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.warmup_send')
    @patch('torch.distributed.get_rank')
    def test_recv_hidden_edge_mode_d_rank_1(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = 'master'
        mode = 'd'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1], [1,0]]
        comm.map_inter_send_recv = [[0,1],[1,0]]
        comm.hidden_size = 64
        comm.hccl_comm_warmup(comm.hidden_size)

        peer_index = 1
        bc_group = comm.groups_inter_send_recv[comm.rank]
        src_rank = comm.d_send_card
        with patch('torch.distributed.irecv') as mock_isend:
            mock_isend.return_value = 1
            comm.recv_hidden(mode=mode, shape=1024)
            if comm.rank == src_rank:
                src = comm.groups_inter_send_recv[comm.rank][peer_index]
                mock_isend.assert_called_once_with(
                    ANY,
                    src=src,
                    group=bc_group
                )
            else:
                mock_isend.assert_not_called()
            if mode == 'p':
                self.assertEqual(comm.ret_p, 1)
            else:
                self.assertEqual(comm.ret_d, 1)

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.warmup_recv')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.warmup_send')
    @patch('torch.distributed.get_rank')
    def test_recv_hidden_cloud_mode_p_rank_0(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = 'slave'
        mode = 'p'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.map_inter_send_recv = [[0,1], [1,0]]
        comm.groups_inter_send_recv = [[0,1], [1,0]]
        comm.hidden_size = 64
        comm.hccl_comm_warmup(comm.hidden_size)

        peer_index = 0
        bc_group = comm.groups_inter_send_recv[comm.rank]
        src_rank = comm.p_send_card if mode == 'p' else comm.d_send_card
        with patch('torch.distributed.irecv') as mock_isend:
            mock_isend.return_value = 1
            comm.recv_hidden(mode=mode, shape=1024)
            if comm.rank == src_rank:
                src = comm.map_inter_send_recv[comm.rank][peer_index]
                mock_isend.assert_called_once_with(
                    ANY,
                    src=src,
                    group=bc_group
                )
            else:
                mock_isend.assert_not_called()
            if mode == 'p':
                self.assertEqual(comm.ret_p, 1)
            else:
                self.assertEqual(comm.ret_d, 1)

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_recv_hidden_cloud_mode_p_rank_1(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = 'slave'
        mode = 'p'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1], [1,0]]
        comm.hidden_size = 64
        with patch('torch.distributed.irecv', return_value="ret") as mock_irecv:
            comm.recv_hidden(mode=mode, shape=1024)
            mock_irecv.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_recv_hidden_cloud_mode_d_rank_0(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = 'slave'
        mode = 'd'
        comm.rank = 0
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.groups_inter_send_recv = [[0,1], [1,0]]
        comm.hidden_size = 64

        comm.ret_d = None
        with patch('torch.distributed.irecv') as mock_isend:
            comm.recv_hidden(mode=mode, shape=1024)
            mock_isend.assert_not_called()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.warmup_recv')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.warmup_send')
    @patch('torch.distributed.get_rank')
    def test_recv_hidden_cloud_mode_d_rank_1(self, *args):
        comm = EdgeCloudDataComm()
        comm.role = 'slave'
        mode = 'd'
        comm.rank = 1
        comm.p_send_card = 0
        comm.d_send_card = 1
        comm.map_inter_send_recv = [[0,1], [1,0]]
        comm.groups_inter_send_recv = [[0,1], [1,0]]
        comm.hidden_size = 64
        comm.hccl_comm_warmup(comm.hidden_size)

        peer_index = 0
        bc_group = comm.groups_inter_send_recv[comm.rank]
        src_rank = comm.d_send_card
        with patch('torch.distributed.irecv') as mock_isend:
            mock_isend.return_value = 1
            comm.recv_hidden(mode=mode, shape=1024)
            if comm.rank == src_rank:
                src = comm.map_inter_send_recv[comm.rank][peer_index]
                mock_isend.assert_called_once_with(
                    ANY,
                    src=src,
                    group=bc_group
                )
            else:
                mock_isend.assert_not_called()
            if mode == 'p':
                self.assertEqual(comm.ret_p, 1)
            else:
                self.assertEqual(comm.ret_d, 1)


    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.ones')
    def test_temp_unset_rank_table(self, *args):
        '''
        测试临时移除RANK_TABLE_FILE环境变量
        '''
        os.environ['RANK_TABLE_FILE'] = '/path/to/rank_table.json'
        comm = EdgeCloudDataComm()
        origin_value = comm.temp_unset_rank_table()
        self.assertEqual(origin_value, '/path/to/rank_table.json')
        self.assertNotIn('RANK_TABLE_FILE', os.environ)

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.ones')
    def test_restore_rank_table(self, *args):
        '''
        测试恢复RANK_TABLE_FILE环境变量
        '''
        comm = EdgeCloudDataComm()
        comm.restore_rank_table('/path/to/rank_table.json')
        self.assertEqual(os.environ.get('RANK_TABLE_FILE'), '/path/to/rank_table.json')
        
        del os.environ['RANK_TABLE_FILE']
        comm.restore_rank_table(None)
        self.assertNotIn('RANK_TABLE_FILE', os.environ)

    @patch('torch.distributed.broadcast')
    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.new_group',return_value=None)
    @patch('torch.distributed.barrier')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.restore_rank_table')
    @patch('atb_llm.utils.layerwise_disaggregated.edge_cloud_data_comm.EdgeCloudDataComm.temp_unset_rank_table')
    @patch('torch.ones')
    def test_init_hccl(self, mock_init_process_group, *args):
        comm = EdgeCloudDataComm()
        data_comm_args = {'edge_ip':"127.0.0.1",'edge_port':"9999",'npuEdgeNum':2,'npuCloudNum':8}

        comm.init_hccl(rank=0, role='master', data_comm_args=data_comm_args)

        comm.init_hccl(rank=0,role='slave',data_comm_args=data_comm_args)

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_warmup_send(self, *args):
        comm = EdgeCloudDataComm()
        comm.groups_inter_send_recv = [0,1]
        comm.map_inter_send_recv = [[0,1], [1,0]]

        comm.rank = 0
        comm.p_send_card = 0
        with patch('torch.distributed.isend') as mock_isend:
            comm.warmup_send(1)
            mock_isend.assert_called_once()

        comm.rank = 1
        comm.d_send_card = 1
        with patch('torch.distributed.isend') as mock_isend:
            comm.warmup_send(1)
            mock_isend.assert_called_once()

    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_warmup_recv(self, *args):
        comm = EdgeCloudDataComm()
        comm.groups_inter_send_recv = [0,1]
        comm.map_inter_send_recv = [[0,1],[1,0]]
        
        comm.rank = 0
        comm.p_recv_card = 0
        with patch("torch.distributed.irecv") as mock_isend:
            comm.warmup_recv(1)
            mock_isend.assert_called_once()
        comm.rank = 1
        comm.d_recv_card = 1
        with patch("torch.distributed.irecv") as mock_isend:
            comm.warmup_recv(1)
            mock_isend.assert_called_once()

    
    @patch.object(EdgeCloudDataComm, '__new__', return_value=object.__new__(EdgeCloudDataComm))
    @patch('torch.npu.stream', side_effect=MockStreamContext)
    @patch('torch.ones')
    def test_broadcast_ctrl(self, *args):
        comm = EdgeCloudDataComm()
        comm.group_intra_broadcast_edge = MagicMock()
        comm.map_intra_broadcast_edge = [0]
        comm.group_intra_broadcast_cloud = MagicMock()
        comm.map_intra_broadcast_cloud = [0]
        
        comm.rank = 0
        comm.role = "master"
        with patch("torch.distributed.broadcast") as mock_broad:
            comm.broadcast_ctrl(switch_flag=1, shape=[10,20])
            mock_broad.assert_called_once()

        comm.rank = 1
        comm.role = "slave"
        with patch("torch.distributed.broadcast") as mock_broad:
            comm.broadcast_ctrl(switch_flag=1, shape=[10,20])
            mock_broad.assert_called_once()

if __name__ == '__main__':
    # 设置测试环境
    os.environp['MINDIE_LLM_HOME'] = '/mock/path'
    
    unittest.main()