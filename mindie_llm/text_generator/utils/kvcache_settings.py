# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
To replace CacheManager
"""
import math
from dataclasses import dataclass
from typing import Tuple
import acl
import torch

from ...modeling.backend_type import BackendType
from ...utils.log.logging import logger, print_log
from ...utils.tensor import npu


# NZ排布的KVCache做了16位对齐
NZ_KV_CACHE_FORMAT = 16
# NZ排布的KVCache int8做了32位对齐
NZ_KV_CACHE_INT8_FORMAT = 32
INT8_BYTES_SIZE = 1
TOTAL_MEMORY = 60 * 1024 * 1024 * 1024
MM_LONG_SEQ_MEMORY = 26 * 1024 * 1024 * 1024
MM_NORMAL_SEQ_MEMORY = 18 * 1024 * 1024 * 1024
MM_LONG_SEQ_TOKENLEN = 4096

torch_dtype_map = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float: "float", torch.int8: "int8"}


@dataclass
class NPUSocInfo:
    __slots__ = ["soc_name", "soc_version", "need_nz"]

    def __post_init__(self):
        self.soc_name = acl.get_soc_name()
        self.need_nz = self.support_nz()

    def support_nz(self) -> bool:
        if self.soc_name is None:
            return False
        nz_name = ("910PremiumA", "910ProA", "910A", "910ProB", "910B")
        for name in nz_name:
            if self.soc_name.upper().endswith(name.upper()):
                return True
        return "310P" in self.soc_name.upper()


def calc_block_mem(model_info, block_size, num_speculative_tokens=None):
    if num_speculative_tokens is None:
        num_speculative_tokens = 0
    total_head_size = model_info.num_kv_heads * model_info.head_size
    # k_total_head_size和v_total_head_size需成对定义
    if model_info.k_head_size > 0 or model_info.v_head_size > 0:
        k_total_head_size = model_info.num_kv_heads * model_info.k_head_size
        v_total_head_size = model_info.num_kv_heads * model_info.v_head_size
    else:
        k_total_head_size = total_head_size
        v_total_head_size = total_head_size
    num_layers = model_info.num_layers + (num_speculative_tokens >= 1)
    per_layer_k_cache_bytes_size = [model_info.data_byte_size for layer_id in range(num_layers)]
    model_mem_size = num_layers * v_total_head_size * model_info.data_byte_size
    if model_info.kvcache_quant_layers:
        for i, kvcache_quant in enumerate(model_info.kvcache_quant_layers):
            if kvcache_quant:
                per_layer_k_cache_bytes_size[i] = INT8_BYTES_SIZE
    for bytes_size in per_layer_k_cache_bytes_size:
        model_mem_size += k_total_head_size * bytes_size
    block_mem_size = model_mem_size * block_size
    return block_mem_size


def calc_npu_mem(block_nums, model_info, block_size, num_speculative_tokens=None):
    block_mem_size = calc_block_mem(model_info, block_size, num_speculative_tokens)
    npu_mem_size = block_nums * block_mem_size
    return npu_mem_size


def gb(mem_size):
    return float(mem_size / (1024 ** 3))


def watch_npu_mem(rank_id, is_multimodal, max_input_len, tag):
    npu.synchronize()
    free_mem, total_mem, _ = acl.rt.get_mem_info(1)
    peak_mem = total_mem - free_mem

    if is_multimodal and total_mem >= TOTAL_MEMORY:
        memory_threshold = MM_LONG_SEQ_MEMORY if max_input_len > MM_LONG_SEQ_TOKENLEN else MM_NORMAL_SEQ_MEMORY
        if free_mem < memory_threshold:
            error_message = (
                f"Warmup failed, because of multimodal model inference out of memory "
                f"when `maxInputTokenLen` set to {max_input_len}. Please try to "
                f"decrease `maxPrefillTokens` in config.json of mindie-service."
            )
            print_log(rank_id, logger.error, error_message)
            raise RuntimeError("NPU out of memory. " + error_message)
    print_log(rank_id, logger.info, f"{tag}, peak mem: {gb(peak_mem):.2f}G, total_mem: {gb(total_mem):.2f}G")
    return total_mem, peak_mem


# IMPORTANT: CacheManager.slots will be moved to BatchCache (InferContext)
class KVCacheSettings:
    __slots__ = [
        "backend_type",
        "num_layers",
        "num_heads",
        "head_size",
        "k_head_size",
        "v_head_size",
        "dtype",
        "data_byte_size",  # not meaningful
        "kvcache_quant_layers",  # need comments
        "block_size",
        "cpu_mem",  # in bytes
        "npu_mem",  # in bytes
        "npu_info",  # NPUSocInfo
        "rank",  # local or global?
        "dtype_str",  # for kvmover use, might need cleanup later, formerly sepd_dtype
        "num_speculative_tokens",
        "mini_block_bytes",  # need clarification
        "k_mini_block_bytes",  # need clarification
        "need_nz",  # bool
        "k_mini_block_bytes_quant",  # ??
        "v_mini_block_bytes",  # ??
        "cache_block_bytes",  # ??
        "num_npu_blocks",
        "num_cpu_blocks",
        "npu_row_bytes",  # ??
        "cpu_row_bytes",  # ??
        "npu_row_bytes_quant",  # ??
        "cpu_row_bytes_quant",  # ??
        "block_shape",
        "k_block_shape",
        "v_block_shape",
        "k_block_quant_shape",
        "is_separated_pd",  # is kvcache for separated PD
    ]

    # need typing
    def __init__(
        self, rank, model_info, cpu_mem, npu_mem, block_size, backend_type, is_separated_pd, num_speculative_tokens=None
    ):
        self.backend_type = backend_type
        self.num_layers = model_info.num_layers
        self.num_heads = model_info.num_kv_heads
        self.head_size = model_info.head_size
        # 为适配k head和v head size不等长的场景，新增k_head_size，v_head_size，k_mini_block_bytes，v_mini_block_bytes，
        # k_block_shape和v_block_shape；默认值和head_size，mini_block_bytes和block_shape保持一致。
        # 兼容等长场景，head_size，mini_block_bytes，block_shape属性含义无修改
        self.k_head_size = model_info.k_head_size
        self.v_head_size = model_info.v_head_size
        self.dtype = model_info.dtype
        self.data_byte_size = model_info.data_byte_size
        self.kvcache_quant_layers = model_info.kvcache_quant_layers

        self.block_size = block_size
        self.cpu_mem = int(cpu_mem * 1024 * 1024 * 1024)
        self.npu_mem = int(npu_mem * 1024 * 1024 * 1024)
        self.npu_info = NPUSocInfo()
        self.rank = rank
        self.npu_info.need_nz |= model_info.enable_nz
        self.dtype_str = self.dtype_to_str(self.backend_type, self.dtype)
        self.num_speculative_tokens = 0 if num_speculative_tokens is None else num_speculative_tokens
        self.num_layers += (self.num_speculative_tokens >= 1)

        self.need_nz = self.npu_info.need_nz or model_info.enable_nz
        if NZ_KV_CACHE_FORMAT == 0:
            raise ZeroDivisionError("NZ_KV_CACHE_FORMAT should not be 0")
        if NZ_KV_CACHE_INT8_FORMAT == 0:
            raise ZeroDivisionError("NZ_KV_CACHE_INT8_FORMAT should not be 0")

        total_head_size, k_total_head_size, v_total_head_size, k_total_head_size_quant = self._cal_kv_total_head_size()
        self.mini_block_bytes = self.block_size * self.data_byte_size * total_head_size
        self.k_mini_block_bytes = self.block_size * self.data_byte_size * k_total_head_size
        self.k_mini_block_bytes_quant = self.block_size * INT8_BYTES_SIZE * k_total_head_size_quant
        self.v_mini_block_bytes = self.block_size * self.data_byte_size * v_total_head_size
        self.cache_block_bytes = self.num_layers * self.v_mini_block_bytes

        per_layer_k_cache_bytes_size = [self.k_mini_block_bytes] * self.num_layers
        if model_info.kvcache_quant_layers:
            for i, kvcache_quant in enumerate(model_info.kvcache_quant_layers):
                if kvcache_quant:
                    per_layer_k_cache_bytes_size[i] = self.k_mini_block_bytes_quant
        for bytes_size in per_layer_k_cache_bytes_size:
            self.cache_block_bytes += bytes_size
        if self.cache_block_bytes == 0:
            raise ZeroDivisionError("self.cache_block_bytes should not be 0")

        self.num_npu_blocks = self.npu_mem // self.cache_block_bytes
        self.num_cpu_blocks = self.cpu_mem // self.cache_block_bytes
        self.npu_row_bytes = self.num_npu_blocks * (self.k_mini_block_bytes + self.v_mini_block_bytes)
        self.cpu_row_bytes = self.num_cpu_blocks * (self.k_mini_block_bytes + self.v_mini_block_bytes)
        self.npu_row_bytes_quant = self.num_npu_blocks * (self.k_mini_block_bytes_quant + self.v_mini_block_bytes)
        self.cpu_row_bytes_quant = self.num_cpu_blocks * (self.k_mini_block_bytes_quant + self.v_mini_block_bytes)
        logger.debug(
            f"block_size:{self.block_size},num_heads:{self.num_heads},head_size:{self.head_size},"
            f"k_head_size:{self.k_head_size},v_head_size:{self.v_head_size},"
            f"num_layers:{self.num_layers}, "
            f"cache_block_bytes:{self.cache_block_bytes},"
            f"k_mini_block_bytes:{self.k_mini_block_bytes},v_mini_block_bytes:{self.v_mini_block_bytes},"
            f"num_cpu_blocks:{self.num_cpu_blocks}, num_npu_blocks:{self.num_npu_blocks}"
        )
        self.block_shape = None
        self.k_block_shape = None
        self.v_block_shape = None
        self.k_block_quant_shape = None
        self.is_separated_pd = is_separated_pd  # use this flag to replace original sped_worker object
        self._cal_set_kv_block_shapes()
        # IMPORTANT : sepd_worker.build() method will be move to generator()

    @staticmethod
    def dtype_to_str(backend_type: BackendType, dtype) -> str:
        if backend_type == BackendType.ATB:
            dtype_map = torch_dtype_map
        else:
            import mindspore

            mindspore_dtype_map = {
                mindspore.float16: "float16",
                mindspore.bfloat16: "bfloat16",
                mindspore.float32: "float",
                mindspore.int8: "int8",
            }
            dtype_map = mindspore_dtype_map
        if dtype not in dtype_map:
            raise Exception("not supported kvcache dtype for ATB backend")
        return dtype_map[dtype]

    def _cal_set_kv_block_shapes(self) -> None:
        self.block_shape = (
            (
                math.ceil(self.num_heads * self.head_size / NZ_KV_CACHE_FORMAT),  # NZ KVCache排布需要16对齐
                self.block_size,
                NZ_KV_CACHE_FORMAT,
            )
            if self.need_nz
            else (self.block_size, self.num_heads, self.head_size)
        )  # ND
        # k_block_shape和v_block_shape需成对定义
        if self.k_head_size > 0 or self.v_head_size > 0:
            self.k_block_shape = (
                (
                    math.ceil(self.num_heads * self.k_head_size / NZ_KV_CACHE_FORMAT),  # NZ KVCache排布需要16对齐
                    self.block_size,
                    NZ_KV_CACHE_FORMAT,
                )
                if self.need_nz
                else (self.block_size, self.num_heads, self.k_head_size)
            )  # ND
            self.v_block_shape = (
                (
                    math.ceil(self.num_heads * self.v_head_size / NZ_KV_CACHE_FORMAT),  # NZ KVCache排布需要16对齐
                    self.block_size,
                    NZ_KV_CACHE_FORMAT,
                )
                if self.need_nz
                else (self.block_size, self.num_heads, self.v_head_size)
            )  # ND
            self.k_block_quant_shape = (
                (
                    math.ceil(self.num_heads * self.k_head_size / NZ_KV_CACHE_INT8_FORMAT),  # NZ KVCache排布需要32对齐
                    self.block_size,
                    NZ_KV_CACHE_INT8_FORMAT,
                )
                if self.need_nz
                else (self.block_size, self.num_heads, self.k_head_size)
            )  # ND
        else:
            self.k_block_shape = self.block_shape
            self.v_block_shape = self.block_shape
            self.k_block_quant_shape = self.block_shape

    # return format: Tuple of total_head_size, k_total_head_size, v_total_head_size, k_total_head_size_quant
    def _cal_kv_total_head_size(self) -> Tuple[int, int, int, int]:
        total_head_size = (
            (math.ceil(self.num_heads * self.head_size / NZ_KV_CACHE_FORMAT) * NZ_KV_CACHE_FORMAT)
            if self.need_nz
            else (self.num_heads * self.head_size)
        )
        # k_total_head_size和v_total_head_size需成对定义
        if self.k_head_size > 0 or self.v_head_size > 0:
            k_total_head_size = (
                (math.ceil(self.num_heads * self.k_head_size / NZ_KV_CACHE_FORMAT) * NZ_KV_CACHE_FORMAT)
                if self.need_nz
                else (self.num_heads * self.k_head_size)
            )
            k_total_head_size_quant = (
                (math.ceil(self.num_heads * self.k_head_size / NZ_KV_CACHE_INT8_FORMAT) * NZ_KV_CACHE_INT8_FORMAT)
                if self.need_nz
                else (self.num_heads * self.k_head_size)
            )

            v_total_head_size = (
                (math.ceil(self.num_heads * self.v_head_size / NZ_KV_CACHE_FORMAT) * NZ_KV_CACHE_FORMAT)
                if self.need_nz
                else (self.num_heads * self.v_head_size)
            )
        else:
            k_total_head_size = total_head_size
            v_total_head_size = total_head_size
            k_total_head_size_quant = total_head_size
        return total_head_size, k_total_head_size, v_total_head_size, k_total_head_size_quant
