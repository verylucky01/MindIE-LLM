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

from mindie_llm.connector.common.input_metadata_composite import InputMetadataComposite
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata


class EdgeCloudInputMetadata:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EdgeCloudInputMetadata, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.input_metadata_composite = [None, None]

    @staticmethod
    def is_get_input_metadata(layerwise_disaggregated_exe_stage: LwdMetadata):
        is_last_layer = (
            layerwise_disaggregated_exe_stage.start_exec_layer == 1 
            and layerwise_disaggregated_exe_stage.end_exec_layer == 1
        )
        is_prefill_with_offset = (
            layerwise_disaggregated_exe_stage.is_prefill 
            and (
                layerwise_disaggregated_exe_stage.start_exec_layer != 0 
                or layerwise_disaggregated_exe_stage.long_seq_start_idx != 0
            )
        )
        return is_last_layer or is_prefill_with_offset
    
    @staticmethod
    def is_storage_input_metadata(layerwise_disaggregated_exe_stage: LwdMetadata):
        stage = layerwise_disaggregated_exe_stage
        is_start_layer_0 = stage.start_exec_layer == 0
        is_end_layer_0 = stage.end_exec_layer == 0

        is_prefill = stage.is_prefill
        is_not_prefill_chunk = not stage.is_long_seq
        is_prefill_offset_0 = stage.long_seq_start_idx == 0
        
        is_valid_prefill = is_prefill and (is_not_prefill_chunk or is_prefill_offset_0)
        return is_start_layer_0 and ((is_end_layer_0 and not is_prefill) or is_valid_prefill)

    def get_input_metadata(self, is_prefill):
        return self.input_metadata_composite[int(is_prefill)]

    def set_input_metadata(self, input_metadata_composite: InputMetadataComposite, is_prefill):
        self.input_metadata_composite[int(is_prefill)] = input_metadata_composite

pd_exec_matadata_instance = EdgeCloudInputMetadata()