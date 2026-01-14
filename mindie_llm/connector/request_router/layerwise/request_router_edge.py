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

import queue
import sys
import time
import itertools
from enum import IntEnum
from pathlib import Path

from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.prof.profiler import span_start, span_end
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.connector.request_router.layerwise.request_router_lwd import RequestRouterLwd, DecisionType, \
    MASTER_ID, LONG_SEQ_LEN_MIN

sys.path.append(str(Path(__file__).parent / "sync"))


class CtrlTypePos(IntEnum):
    DECISION_TYPE = 0
    SHAPE_START = 1
    SHAPE_END = 2
    SEQ_LEN = 3
    BATCH_SIZE = 4
    MAX_NUM = 5


class RequestRouterEdge(RequestRouterLwd):
    def __init__(self):
        self.prefill_first = False  
        self.decode_first = False 
        self.prefill_first_finish = False
        self.decode_first_finish = False

        '''
        Wait for the next Decode with timing; 
        non-empty indicates the previous execution was a D-end task.
        '''
        self.force_wait_d_time = None
        self.decode_tcp_recv_time = None
        
        self.prefill_chunk_metadata_queue = queue.Queue()
        self.is_last_do_chunk_prefill = False
        self.wait_decode_comm = False

        self.process_func = {
            DecisionType.DO_PREFILL_FIRST: self.do_prefill_first,
            DecisionType.DO_PREFILL_LAST: self.do_prefill_last,
            DecisionType.DO_DECODE_FIRST: self.do_decode_first,
            DecisionType.DO_DECODE_LAST: self.do_decode_last,
            DecisionType.DO_CLEAN_UP: self.do_clean_up,
            DecisionType.DO_CLEAN_EOS: self.do_clean_eos,
        }
        
        super().__init__()

    def initialize_diff(self, model_config, models_config_dict):
        is_producer = True if self.rank == MASTER_ID else False
        card_num = models_config_dict.get('layerwiseDisaggregatedMasterDeviceNum', 2)
        self.mem_manager.initialize(is_producer, card_num - 1)

        logger.info(f"[layerwiseDisaggregated] edge initliaze ok rank:{self.rank}, is_producer:{is_producer}, "
            f"card_num:{card_num}")

    def do_prefill_first(self):
        prof = span_start("Prefill_first")
        logger.info(f"[layerwiseDisaggregated] execute prefill_first before, rank:{self.rank}.")
        while self.prefill_request is None:
            self.get_all_request()
        logger.info(f"[layerwiseDisaggregated] execute prefill_first end, rank:{self.rank}.")
        self.router_impl.execute(self.prefill_request)
        if not self.is_long_seq:
            self.prefill_first = False
            self.is_last_do_chunk_prefill = False
        else:
            self.prefill_chunk_variable_policy()
        span_end(prof)

    def do_prefill_last(self):
        prof = span_start("Prefill_last")
        logger.info(f"[layerwiseDisaggregated] execute prefill_last before, rank:{self.rank}.")
        while self.prefill_request is None:
            self.get_all_request()
        self.router_impl.execute(self.prefill_request)
        logger.info(f"[layerwiseDisaggregated] execute prefill_last end, rank:{self.rank}.")
        if not self.is_long_seq:
            self.prefill_comm_finish = False
            self.prefill_request = None
            self.prefill_seq_len = 0
            self.prefill_batch_size = 0
            self.is_last_do_chunk_prefill = False
        else:
            self.prefill_chunk_variable_policy()
        span_end(prof)

    def do_decode_first(self):
        prof = span_start("Decode_first")
        logger.info(f"[layerwiseDisaggregated] execute decode_first before, rank:{self.rank}.")
        while self.decode_request is None:
            self.get_all_request()
        self.router_impl.execute(self.decode_request)
        logger.info(f"[layerwiseDisaggregated] execute decode_first end, rank:{self.rank}.")
        self.decode_first = False
        self.is_last_do_chunk_prefill = False
        span_end(prof)
    
    def do_decode_last(self):
        prof = span_start("Decode_last")
        logger.info(f"[layerwiseDisaggregated] execute decode_last before, rank:{self.rank}.")
        while self.decode_request is None:
            self.get_all_request()
        self.router_impl.execute(self.decode_request)
        if self.rank == MASTER_ID:
            self.force_wait_d_time = time.time()     # Wait for the next Decode with timing.
        logger.info(f"[layerwiseDisaggregated] execute decode_last end, rank:{self.rank}.")
        self.decode_comm_finish = False
        self.ctrl_comm.decode_comm_finish = False
        self.decode_request = None
        self.is_last_do_chunk_prefill = False
        span_end(prof)

    def do_clean_eos(self):
        while self.clean_eos_queue.empty():
            time.sleep(0.001)
            self.get_all_request()

        self.clean_eos_queue.get()  # 边侧推出eos会自动清理cache, 下eos请求下给云侧清理cache使用
        logger.info(f"[layerwiseDisaggregated][python thread: infer] text generator clean eos, rank{self.rank}.")

    def recv_decode(self):
        self.ctrl_comm.recv_decode()
        self.decode_comm_finish = self.ctrl_comm.decode_comm_finish
        logger.info(f"[layerwiseDisaggregated] decode_comm_finish = {self.decode_comm_finish}")
        if self.decode_comm_finish:
            self.decode_shape = self.ctrl_comm.parse_shape(self.ctrl_comm.decode_recv_msg)
            logger.info("[layerwiseDisaggregated] edge recv_decode, comm finish.")

    def check_10ms_for_next_decode(self):
        if self.force_wait_d_time is None:
            return
        self.wait_d_time_gap = time.time() - self.force_wait_d_time
        if self.wait_d_time_gap > 0.01:
            logger.info(f"[layerwiseDisaggregated] Force wait decode state exit, "
                f"wait time: {self.wait_d_time_gap * 1000} ms.")
            self.force_wait_d_time = None
            self.wait_d_time_gap = 0

    def calc_prefill_priority_decision_type(self):
        if self.force_wait_d_time and self.decode_request and self.decode_first:
            self.decision_type = DecisionType.DO_DECODE_FIRST
            self.force_wait_d_time = None
            logger.info(f"[layerwiseDisaggregated] Force to do decode first, "
                f"wait time: {self.wait_d_time_gap * 1000} ms.")
            self.wait_d_time_gap = 0
        elif self.force_wait_d_time:
            self.decision_type = DecisionType.WAIT_DECODE
        elif self.prefill_request and self.prefill_first:
            self.decision_type = DecisionType.DO_PREFILL_FIRST
        elif self.prefill_request and self.prefill_comm_finish:
            self.decision_type = DecisionType.DO_PREFILL_LAST
        elif self.decode_request and self.decode_first:
            self.decision_type = DecisionType.DO_DECODE_FIRST
        elif self.decode_request and self.decode_comm_finish:
            self.decision_type = DecisionType.DO_DECODE_LAST

    def calc_decode_priority_decision_type(self):
        if self.decode_request and self.decode_first:
            self.decision_type = DecisionType.DO_DECODE_FIRST
        elif self.decode_request and self.decode_comm_finish:
            self.decision_type = DecisionType.DO_DECODE_LAST
        elif self.decode_request and self.wait_decode_comm:
            self.decision_type = DecisionType.WAIT_COMM
        elif self.prefill_request and self.prefill_first:
            self.decision_type = DecisionType.DO_PREFILL_FIRST
        elif self.prefill_request and self.prefill_comm_finish:
            self.decision_type = DecisionType.DO_PREFILL_LAST

    def decision_do_clean_eos_type(self):
        if not self.clean_eos_queue.empty():
            self.decision_type = DecisionType.DO_CLEAN_EOS
            return True

        return False

    def calc_decision_type(self):
        self.check_10ms_for_next_decode()
        self.decision_type = DecisionType.WAIT_COMM

        if self.decision_do_clean_eos_type():
            return

        if self.decision_do_clean_up_type():
            return
        '''
        If the previous task was a decode-last and a new decode-first task arrives within 10ms, 
        execute the decode-first task.
        '''
        if not self.is_last_do_chunk_prefill:
            self.calc_prefill_priority_decision_type()
        else:
            self.calc_decode_priority_decision_type()
    
    def prepare_chunk_prefill_metadata_queue(self, prefill_seq_len, prefill_chunk_num):
        if self.prefill_chunk_metadata_queue.qsize() > 0:
            return
        average_prefill_chunk_seq_len = int(prefill_seq_len / prefill_chunk_num)
        mod = prefill_seq_len % prefill_chunk_num
        prefill_chunk_policy: list = ([0] + [average_prefill_chunk_seq_len] * prefill_chunk_num if mod == 0 else [0] + 
            [average_prefill_chunk_seq_len + 1] * mod + [average_prefill_chunk_seq_len] * (prefill_chunk_num - mod)
        )
        prefill_chunk_policy = list(itertools.accumulate(prefill_chunk_policy))
        logger.info(f"[layerwiseDisaggregated] prefill_chunk_num is {prefill_chunk_num}, prefill_chunk_policy is "
                    f"{prefill_chunk_policy}, rank {self.rank}")

        prefill_seq_len = self.prefill_seq_len
        # 实现逻辑为: 将一个P切成n段, 执行顺序为: P0首 -> P1首 P0尾 P2首 P1尾 ... Pn-1首 Pn-2尾 -> Pn-1尾
        for i in range(prefill_chunk_num):
            if i == 0:
                start_offset = 0
                end_offset = prefill_chunk_policy[i + 1]
                metadata = LwdMetadata(0, 0, False, True, 1, 1, 0, True, start_offset, end_offset, 0, prefill_seq_len)
                self.prefill_chunk_metadata_queue.put(metadata)

            if i > 0 and i <= prefill_chunk_num - 1:
                start_offset = prefill_chunk_policy[i]
                end_offset = prefill_chunk_policy[i + 1]
                metadata = LwdMetadata(0, 0, False, True, 1, 1, 0, True, start_offset, end_offset, 0, prefill_seq_len)
                self.prefill_chunk_metadata_queue.put(metadata)

                start_offset = prefill_chunk_policy[i - 1]
                end_offset = prefill_chunk_policy[i]
                metadata = LwdMetadata(1, 1, False, True, 1, 1, 0, True, start_offset, end_offset, 0, prefill_seq_len)
                self.prefill_chunk_metadata_queue.put(metadata)

            if i == prefill_chunk_num - 1:
                start_offset = prefill_chunk_policy[i]
                end_offset = prefill_chunk_policy[i + 1]
                metadata = LwdMetadata(1, 1, True, True, 1, 1, 0, True, start_offset, end_offset, 0, prefill_seq_len)
                self.prefill_chunk_metadata_queue.put(metadata)

    # prefill请求执行结束之后, 处理下一次调度需要的变量           
    def prefill_chunk_variable_policy(self):
        self.is_last_do_chunk_prefill = True
        metadata = lwd_metadata_manager.get_metadata()
        start_exec_layer = metadata.start_exec_layer
        end_exec_layer = metadata.end_exec_layer
        long_seq_start_idx = metadata.long_seq_start_idx
        end_of_generate_token = metadata.end_of_generate_token
        if start_exec_layer == 0 and end_exec_layer == 0 and long_seq_start_idx == 0:   # 首trunk
            self.prefill_first = True
            self.wait_decode_comm = True
        elif self.prefill_chunk_metadata_queue.qsize() == 1:    # 倒数第二个trunk
            self.prefill_first = False
            self.prefill_comm_finish = False
        elif end_of_generate_token:    # 最后一个trunk
            self.prefill_comm_finish = False
            self.prefill_request = None
            self.prefill_seq_len = 0
            self.prefill_batch_size = 0
            self.is_long_seq = False
            self.prefill_chunk_num = 1
            self.is_last_do_chunk_prefill = False
        elif start_exec_layer == 0 and end_exec_layer == 0:     # P首清除变量
            self.prefill_first = False
        elif start_exec_layer == 1 and end_exec_layer == 1:     # P尾清除变量
            self.prefill_first = True
            self.prefill_comm_finish = False

    def arrange_exec_stage(self):
        if self.decision_type == DecisionType.DO_DECODE_FIRST:
            metadata = LwdMetadata(0, 0, False, False, 1, 1, 0, False, 0, 0, 0, 0)
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(f"[layerwiseDisaggregated] exec stage DO_DECODE_FIRST metadata{metadata}, rank{self.rank}")
        elif self.decision_type == DecisionType.DO_DECODE_LAST:
            metadata = LwdMetadata(1, 1, True, False, 1, 1, 0, False, 0, 0, 0, 0)
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(f"[layerwiseDisaggregated] exec stage DO_DECODE_LAST metadata{metadata}, rank{self.rank}")
        elif self.decision_type == DecisionType.DO_PREFILL_FIRST:
            if not self.is_long_seq:
                metadata = LwdMetadata(0, 0, False, True, 1, 1, 0, False, 0, 0, self.prefill_seq_len)
                lwd_metadata_manager.set_metadata(metadata)
                logger.info(f"[layerwiseDisaggregated] exec stage DO_PREFILL_FIRST metadata{metadata}, rank{self.rank}")
            else:
                metadata = self.prefill_chunk_metadata_queue.get(block=False)
                lwd_metadata_manager.set_metadata(metadata)
                logger.info(f"[layerwiseDisaggregated] exec stage DO_PREFILL_FIRST(chunk) metadata{metadata}, "
                    f"rank{self.rank}")
        elif self.decision_type == DecisionType.DO_PREFILL_LAST:
            if not self.is_long_seq:
                metadata = LwdMetadata(1, 1, True, True, 1, 1, 0, False, 0, 0, self.prefill_seq_len)
                lwd_metadata_manager.set_metadata(metadata)
                logger.info(f"[layerwiseDisaggregated] exec stage DO_PREFILL_LAST metadata{metadata}, rank{self.rank}")
            else:
                metadata = self.prefill_chunk_metadata_queue.get(block=False)
                lwd_metadata_manager.set_metadata(metadata)
                logger.info(f"[layerwiseDisaggregated] exec stage DO_PREFILL_LAST(chunk) metadata{metadata}, "
                    f"rank{self.rank}")

    def broadcast_decision_type(self):
        if self.process_func.get(self.decision_type) is None:   # 无需广播的决策
            return

        ctrl_tensor = [0] * CtrlTypePos.MAX_NUM
        ctrl_tensor[CtrlTypePos.DECISION_TYPE] = self.decision_type
        ctrl_tensor[CtrlTypePos.SHAPE_START] = -1
        ctrl_tensor[CtrlTypePos.SHAPE_END] = -1
        ctrl_tensor[CtrlTypePos.SEQ_LEN] = self.prefill_seq_len
        ctrl_tensor[CtrlTypePos.BATCH_SIZE] = self.prefill_batch_size
        self.mem_manager.write_list_memory(ctrl_tensor)

    def recv_decision_type(self):
        ctrl_tensor = self.mem_manager.read_list_memory(self.rank)
        logger.info(f"[layerwiseDisaggregated] recv_decision_type ctrl_tensor {ctrl_tensor}, rank{self.rank}")
        if ctrl_tensor is None:
            self.decision_type = DecisionType.WAIT_COMM
            return

        self.decision_type = DecisionType(int(ctrl_tensor[CtrlTypePos.DECISION_TYPE]))
        self.prefill_seq_len = int(ctrl_tensor[CtrlTypePos.SEQ_LEN])
        self.prefill_batch_size = int(ctrl_tensor[CtrlTypePos.BATCH_SIZE])
        if self.prefill_seq_len > LONG_SEQ_LEN_MIN and self.prefill_batch_size == 1:
            self.is_long_seq = True
            self.prefill_chunk_num = self.prefill_chunk_instance.map_prefill_chunk_num(self.prefill_seq_len)
            self.prepare_chunk_prefill_metadata_queue(self.prefill_seq_len, self.prefill_chunk_num)

        shape = ctrl_tensor[CtrlTypePos.SHAPE_START:CtrlTypePos.SHAPE_END + 1]
        if self.decision_type == DecisionType.DO_DECODE_LAST:
            self.ctrl_comm.decode_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        elif self.decision_type == DecisionType.DO_PREFILL_LAST:
            self.ctrl_comm.prefill_recv_msg = self.ctrl_comm.shape_to_msg(shape)

    def set_pd_curr_request(self):
        if self.prefill_request is None and not self.prefill_queue.empty():
            self.prefill_request = self.prefill_queue.get()
            self.prefill_first = True
            self.prefill_seq_len = self.calc_seq_len(self.prefill_request)
            self.prefill_batch_size = self.calc_batch_size(self.prefill_request)
            if self.prefill_seq_len > LONG_SEQ_LEN_MIN and self.prefill_batch_size == 1 and self.rank == MASTER_ID:
                self.is_long_seq = True
                self.prefill_chunk_num = self.prefill_chunk_instance.map_prefill_chunk_num(self.prefill_seq_len)
                self.prepare_chunk_prefill_metadata_queue(self.prefill_seq_len, self.prefill_chunk_num)
        if self.decode_request is None and not self.decode_queue.empty():
            self.decode_request = self.decode_queue.get()
            self.decode_first = True
