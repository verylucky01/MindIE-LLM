# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import pkgutil
import warnings
import torch

__all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))

# 导入so 和 python
so_directory = os.path.dirname(os.path.abspath(__file__))

os.environ['ASCEND_CUSTOM_OPP_PATH'] = so_directory + '/opp/vendors/customize/' + ':' + os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')
os.environ['LD_LIBRARY_PATH'] = so_directory + '/opp/vendors/customize/op_api/lib/' + ':' + os.environ.get('LD_LIBRARY_PATH', '')
# 遍历目录，加载所有 .so 文件
for filename in os.listdir(so_directory):
    if filename.endswith(".so"):
        filepath = os.path.join(so_directory, filename)
        torch.ops.load_library(filepath)

# Ensure that the torch and torch_npu has been successfully imported to avoid subsequent mount operation failures
import torch
import torch_npu

custom_ops_module = getattr(torch.ops, 'custom', None)

if custom_ops_module is not None:
    for op_name in dir(custom_ops_module):
        if op_name.startswith('_'):
            # skip built-in method, such as __name__, __doc__
            continue

        # get custom ops and set to torch_npu
        custom_op_func = getattr(custom_ops_module, op_name)
        setattr(torch_npu, op_name, custom_op_func)

else:
    warn_msg = "torch.ops.custom module is not found, mount custom ops to torch_npu failed." \
               "Calling by torch_npu.xxx for custom ops is unsupported, please use torch.ops.custom.xxx."
    warnings.warn(warn_msg)
    warnings.filterwarnings("ignore", message=warn_msg)
