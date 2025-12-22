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

import os
import queue
import threading

import torch
import torch_npu

from atb_llm.utils.log import logger

EDGE = "master"
CLOUD = "slave"
HCCL = 'hccl'

P_CARD = 0
D_CARD = 1

SEQ_LEN = 135 * 1024 + 1024
BATCH_LEN = 1024


class EdgeCloudDataComm():
    def __init__(self, dtype=torch.bfloat16):
        self.role = None
        self.edge_ip = None
        self.edge_port = None
        self.rank = None
        self.dist_init = False
        self.dtype = dtype
        self.edge_ranks_num = None
        self.cloud_ranks_num = None
        self.init_finish = False

        self.group_intra_broadcast_edge = None
        self.map_intra_broadcast_edge = []
        self.group_intra_broadcast_cloud = None
        self.map_intra_broadcast_cloud = []
        self.groups_inter_send_recv = None
        self.map_inter_send_recv = []

        self.send_stream = torch.npu.Stream()
        self.recv_stream = torch.npu.Stream()

        self.p_send_card = P_CARD
        self.d_send_card = D_CARD
        self.p_recv_card = P_CARD
        self.d_recv_card = D_CARD

        self.hidden_size = 7168 # No specific requirements; considering general model settings, use 7168.

        self.out_hidden_p = None
        self.out_hidden_d = None
        self.target_p = None
        self.target_d = None
        self.ret_p = None
        self.ret_d = None

        self.prefill_seq_len_queue = queue.Queue()
        self.decode_batch_size_queue = queue.Queue()
        self.p_shape = None
        self.d_shape = None

        self.lock = threading.Lock()
        self.flag_pre_recv = True

        self.need_set_decode_device = False
        self.need_set_prefill_device = False
        self.set_decode_device_done = False
        self.set_prefill_device_done = False

    @staticmethod
    def temp_unset_rank_table():
        original_value = os.environ.get('RANK_TABLE_FILE')
        os.environ.pop('RANK_TABLE_FILE', None)
        logger.info("[layerwiseDisaggregated] remove RANK_TABLE_FILE ENV")
        return original_value

    @staticmethod
    def restore_rank_table(original_value):
        if original_value is not None:
            os.environ['RANK_TABLE_FILE'] = original_value
            logger.info(f"[layerwiseDisaggregated] RECOVER RANK_TABLE_FILE: {original_value}")
        else:
            os.environ.pop('RANK_TABLE_FILE', None)
            logger.info("[layerwiseDisaggregated] CONFIRM RANK_TABLE_FILE is None")

    def init_hccl(self, rank=None, role=None, data_comm_args=None):
        self.role = role
        self.edge_ip = data_comm_args['edge_ip']
        self.edge_port = data_comm_args['edge_port']
        self.rank = rank

        self.edge_ranks_num = data_comm_args['npuEdgeNum']
        self.cloud_ranks_num = data_comm_args['npuCloudNum']

        original_file = EdgeCloudDataComm.temp_unset_rank_table()
        os.environ['MASTER_ADDR'] = self.edge_ip
        os.environ['MASTER_PORT'] = str(self.edge_port)
        os.environ['WORLD_SIZE'] = str(self.edge_ranks_num + self.cloud_ranks_num)

        if self.role == EDGE:
            tmp_rank = 0
        else:
            tmp_rank = self.edge_ranks_num
        os.environ['RANK'] = str(self.rank + tmp_rank)
        torch.distributed.init_process_group(backend=HCCL, init_method='env://')
        logger.info(f"[layerwiseDisaggregated] EdgeCloudDataComm, rank {str(self.rank + tmp_rank)} init_process_group")

        # Definition of edge-cloud broadcast group
        self.group_intra_broadcast_edge = torch.distributed.new_group(ranks=list(range(0, self.edge_ranks_num)),
                                                                backend=HCCL)
        self.map_intra_broadcast_edge = list(range(0, self.edge_ranks_num))
        self.group_intra_broadcast_cloud = torch.distributed.new_group(
            ranks=list(range(self.edge_ranks_num, self.edge_ranks_num + self.cloud_ranks_num)), backend=HCCL)
        self.map_intra_broadcast_cloud = list(range(self.edge_ranks_num, self.edge_ranks_num + self.cloud_ranks_num))
        self.dist_init = True
        logger.info(f"[layerwiseDisaggregated] EdgeCloudDataComm init braodcast groups: \
            {self.map_intra_broadcast_edge, self.map_intra_broadcast_cloud}")

        # Definition of inter-node send-recv group
        self.groups_inter_send_recv = []
        for i in range(self.edge_ranks_num):
            self.groups_inter_send_recv.append(
                torch.distributed.new_group(ranks=[i, i + self.edge_ranks_num], backend=HCCL))
            self.map_inter_send_recv.append([i, i + self.edge_ranks_num])
        logger.info(f"[layerwiseDisaggregated] EdgeCloudDataComm init map_inter_send_recv: {self.map_inter_send_recv}")

        torch.distributed.barrier()
        data = torch.tensor([0], dtype=torch.float16, device='npu')
        torch.distributed.broadcast(data, src=0)
        torch_npu.npu.synchronize()
        EdgeCloudDataComm.restore_rank_table(original_file)
        logger.info("[layerwiseDisaggregated] EdgeCloudDataComm: cloud broadcast group init success")

        self.init_finish = True
        # No inter-node communication warmup is performed here; it will be conducted prior to model-level computation.

    def hccl_comm_warmup(self, hidden_size):
        if hidden_size and hidden_size != self.hidden_size:
            self.hidden_size = hidden_size
        self.out_hidden_p = torch.ones((SEQ_LEN, self.hidden_size), dtype=self.dtype, device='npu')
        self.out_hidden_d = torch.ones((BATCH_LEN, self.hidden_size), dtype=self.dtype, device='npu')
        if self.role == EDGE:
            self.warmup_send(1)
            self.warmup_recv(1)
        else:
            self.warmup_recv(0)
            self.warmup_send(0)
        logger.info(f"[layerwiseDisaggregated] EdgeCloudDataComm Warmup send-recv group finish.\
            {torch.distributed.get_rank()} {self.rank}")

    def warmup_recv(self, peer_index):
        with torch.npu.stream(self.recv_stream):
            if self.rank == self.p_recv_card:
                ret = torch.distributed.irecv(torch.ones((4096, self.hidden_size), dtype=self.dtype, device='npu'),
                                              group=self.groups_inter_send_recv[self.rank],
                                              src=self.map_inter_send_recv[self.rank][peer_index])
                ret.wait()
                torch_npu.npu.synchronize()
            if self.rank == self.d_recv_card:
                ret = torch.distributed.irecv(torch.ones((40, self.hidden_size), dtype=self.dtype, device='npu'),
                                              group=self.groups_inter_send_recv[self.rank],
                                              src=self.map_inter_send_recv[self.rank][peer_index])
                ret.wait()
                torch_npu.npu.synchronize()

    def warmup_send(self, peer_index):
        with torch.npu.stream(self.send_stream):
            if self.rank == self.p_send_card:
                ret = torch.distributed.isend(torch.ones((4096, self.hidden_size), dtype=self.dtype, device='npu'),
                                              group=self.groups_inter_send_recv[self.rank],
                                              dst=self.map_inter_send_recv[self.rank][peer_index])
                ret.wait()
                torch_npu.npu.synchronize()
            if self.rank == self.d_send_card:
                ret = torch.distributed.isend(torch.ones((40, self.hidden_size), dtype=self.dtype, device='npu'),
                                              group=self.groups_inter_send_recv[self.rank],
                                              dst=self.map_inter_send_recv[self.rank][peer_index])
                ret.wait()
                torch_npu.npu.synchronize()

    def broadcast_ctrl(self, switch_flag, shape):
        if self.rank == 0:
            ctrl_tensor = torch.tensor([switch_flag] + shape + [0], dtype=torch.int64, device='npu')
        else:
            ctrl_tensor = torch.tensor([switch_flag, -1, -1, -1], dtype=torch.int64, device='npu')

        if self.role == EDGE:
            torch.distributed.broadcast(ctrl_tensor, group=self.group_intra_broadcast_edge,
                                        src=self.map_intra_broadcast_edge[0])
        else:
            torch.distributed.broadcast(ctrl_tensor, group=self.group_intra_broadcast_cloud,
                                        src=self.map_intra_broadcast_cloud[0])

        return ctrl_tensor

    def broadcast_hidden(self, bc_tensor, shape: int, mode: str):
        src_rank = self.p_recv_card if mode == 'p' else self.d_recv_card
        bc_group = self.group_intra_broadcast_edge if self.role == EDGE else self.group_intra_broadcast_cloud
        bc_group_map = self.map_intra_broadcast_edge if self.role == EDGE else self.map_intra_broadcast_cloud
        if bc_tensor is None:
            if mode == 'p':
                self.target_p = self.out_hidden_p[:shape, :]
                torch.distributed.broadcast(self.target_p, src=bc_group_map[src_rank], group=bc_group)
                return self.target_p
            else:
                self.target_d = self.out_hidden_d[:shape, :]
                torch.distributed.broadcast(self.target_d, src=bc_group_map[src_rank], group=bc_group)
                return self.target_d
        else:
            torch.distributed.broadcast(bc_tensor, src=bc_group_map[src_rank], group=bc_group)
            return bc_tensor

    def send_hidden(self, mode: str, out_tensor):
        peer_index = 1 if self.role == EDGE else 0
        src_rank = self.p_send_card if mode == 'p' else self.d_send_card

        if self.rank == src_rank:
            self.send_stream.wait_stream(torch.npu.default_stream())
            with torch.npu.stream(self.send_stream):
                _ = torch.distributed.isend(tensor=out_tensor, dst=self.map_inter_send_recv[self.rank][peer_index],
                                            group=self.groups_inter_send_recv[self.rank])

    def recv_hidden(self, mode: str, shape: int):
        peer_index = 1 if self.role == EDGE else 0
        src_rank = self.p_recv_card if mode == 'p' else self.d_recv_card

        if self.rank == src_rank:
            if mode == 'p':
                if self.role == CLOUD and not self.set_prefill_device_done and self.need_set_prefill_device:
                    torch.npu.set_device(torch.device(f"npu:{P_CARD}"))
                    self.set_prefill_device_done = True
                self.target_p = self.out_hidden_p[:shape, :]
                with torch.npu.stream(self.recv_stream):
                    ret = torch.distributed.irecv(self.target_p, src=self.map_inter_send_recv[self.rank][peer_index],
                                                  group=self.groups_inter_send_recv[self.rank])
                self.ret_p = ret
            else:
                if self.role == CLOUD and not self.set_decode_device_done and self.need_set_decode_device:
                    torch.npu.set_device(torch.device(f"npu:{D_CARD}"))
                    self.set_decode_device_done = True
                self.target_d = self.out_hidden_d[:shape, :]
                with torch.npu.stream(self.recv_stream):
                    ret = torch.distributed.irecv(self.target_d, src=self.map_inter_send_recv[self.rank][peer_index],
                                                  group=self.groups_inter_send_recv[self.rank])
                self.ret_d = ret

    def data_wait_after_recv(self, mode: str):
        src_rank = self.p_recv_card if mode == 'p' else self.d_recv_card

        if self.rank == src_rank:
            if mode == 'p':
                self.ret_p.wait()
                torch.npu.default_stream().wait_stream(self.recv_stream)
                return self.target_p
            else:
                self.ret_d.wait()
                torch.npu.default_stream().wait_stream(self.recv_stream)
                return self.target_d

        return None
