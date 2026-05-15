# SPDX-FileCopyrightText: Copyright contributors to the vllm-project
# SPDX-License-Identifier: Apache-2.0
# Part of this file implemented based on vllm-project.
#
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import triton
import triton.language as tl


@triton.jit
def muls_add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    scale,  # Scale factor.
    n_elements,  # Size of the vector.
    n_blocks,  # Total number of blocks.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_id in range(pid, n_blocks, num_programs):
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x * scale + y
        tl.store(output_ptr + offsets, output, mask=mask)


def muls_add_triton(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    assert x.shape == y.shape, "Input tensors must have the same shape."
    hidden_size = x.shape[-1]

    n_elements = x.numel()
    output = torch.empty_like(x)

    # Define block size
    BLOCK_SIZE = max(hidden_size // 2, 1024)

    # Calculate the number of programs to launch
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    # Launch the Triton kernel
    muls_add_kernel[grid](
        x,
        y,
        output,
        scale,
        n_elements,
        num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
