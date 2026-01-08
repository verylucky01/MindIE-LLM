#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from enum import Enum


class ErrorCode(str, Enum):
    #ATB_MODELS
    ATB_MODELS_PARAM_OUT_OF_RANGE = "MIE05E000000"
    ATB_MODELS_MODEL_PARAM_JSON_INVALID = "MIE05E000001"
    ATB_MODELS_EXECUTION_FAILURE = "MIE05E000002"
    ATB_MODELS_PARAM_INVALID = "MIE05E000003"
    ATB_MODELS_INTERNAL_ERROR = "MIE05E000004"

    def __str__(self):
        return self.value

    


    
