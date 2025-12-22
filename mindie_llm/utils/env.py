#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
from dataclasses import dataclass, field
import pathlib


def get_benchmark_filepath():
    home_path = os.getenv("MINDIE_LLM_HOME_PATH") if os.getenv("MINDIE_LLM_HOME_PATH") is not None else ""
    return os.getenv(
        "MINDIE_LLM_BENCHMARK_FILEPATH",
        os.path.join(
            home_path,
            "logs/benchmark.jsonl"
        )
    )


def get_benchmark_reserving_ratio():
    return float(
        os.getenv("MINDIE_LLM_BENCHMARK_RESERVING_RATIO", "0.1")
    )


def get_log_file_level():
    return str(
        os.getenv(
            "MINDIE_LOG_LEVEL", os.getenv(
                "MIES_PYTHON_LOG_LEVEL", os.getenv("MINDIE_LLM_PYTHON_LOG_LEVEL", "INFO")
            )
        )
    )


def get_log_to_file():
    return str(
        os.getenv(
            "MINDIE_LOG_TO_FILE", os.getenv(
                "MIES_PYTHON_LOG_TO_FILE", os.getenv("MINDIE_LLM_PYTHON_LOG_TO_FILE", "1")
            )
        )
    )


def get_log_file_path():
    return str(
        os.getenv(
            "MINDIE_LOG_PATH", ""
        )
    )


def get_log_to_stdout():
    return str(
        os.getenv(
            "MINDIE_LOG_TO_STDOUT",
            os.getenv(
                "MIES_PYTHON_LOG_TO_STDOUT",
                os.getenv("MINDIE_LLM_PYTHON_LOG_TO_STDOUT", "1")
            )
        )
    )


def get_log_file_maxsize():
    return int(
        os.getenv('MINDIE_LLM_PYTHON_LOG_MAXSIZE', "20971520")
    )


def get_use_mb_swapper():
    value = os.getenv(
        "MINDIE_LLM_USE_MB_SWAPPER", os.getenv("MIES_USE_MB_SWAPPER", "0")
    )
    if value not in ["0", "1"]:
        raise ValueError("MINDIE_LLM_USE_MB_SWAPPER and MIES_USE_MB_SWAPPER should be 0 or 1")
    if value == "1":
        return True
    else:
        return False


def get_performance_prefix_tree():
    return os.getenv(
        "PERFORMANCE_PREFIX_TREE_ENABLE", "0"
    ) == "1"


def get_visible_devices():
    value = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
    if value is None or value.strip() == "":
        return None
    try:
        return list(map(int, value.split(',')))
    except ValueError as e:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES should be in format "
                         "{device_id},{device_id},...,{device_id}") from e


@dataclass
class EnvVar:
    """
    环境变量
    """
    # 模型运行时动态申请现存池大小（单位：GB）
    reserved_memory_gb: int = field(default_factory=lambda: _parse_int_env("RESERVED_MEMORY_GB", 0))

    # 使用哪些卡
    visible_devices: list[int] | None = field(default_factory=get_visible_devices)
    # 是否绑核
    bind_cpu: bool = field(default_factory=lambda: os.getenv("BIND_CPU", "1") == "1")

    memory_fraction: float = field(default_factory=lambda: float(os.getenv("NPU_MEMORY_FRACTION", "0.8")))

    # 是否记录服务化benchmark所需的性能数据
    benchmark_enable: bool = field(default_factory=lambda: os.getenv("MINDIE_LLM_BENCHMARK_ENABLE", "0") == "1")
    benchmark_enable_async: bool = field(default_factory=lambda: os.getenv("MINDIE_LLM_BENCHMARK_ENABLE", "0") == "2")
    benchmark_filepath: str = field(default_factory=get_benchmark_filepath)
    benchmark_reserving_ratio: float = field(default_factory=get_benchmark_reserving_ratio)
    # 日志级别
    log_file_level: str = field(default_factory=get_log_file_level)
    # 日志是否打印
    log_to_file: str = field(default_factory=get_log_to_file)
    # 日志路径
    log_file_path: str = field(default_factory=get_log_file_path)
    # 日志是否输出到命令行
    log_to_stdout: str = field(default_factory=get_log_to_stdout)
    # 日志最大大小
    log_file_maxsize: int = field(default_factory=get_log_file_maxsize)
    # 日志最大数量
    log_file_maxnum: int = field(default_factory=lambda: _parse_int_env('MINDIE_LLM_PYTHON_LOG_MAXNUM', 10))
    # 日志可选内容
    log_verbose: str = field(default_factory=lambda: str(os.getenv("MINDIE_LOG_VERBOSE", "1")))

    # 是否开启基于memory bridge 的swapper优化
    use_mb_swapper: bool = field(default_factory=get_use_mb_swapper)

    # 选择后处理加速模式
    speed_mode_type: int = field(default_factory=lambda: _parse_int_env("POST_PROCESSING_SPEED_MODE_TYPE", 0))

    rank: int = field(default_factory=lambda: _parse_int_env("RANK", 0))
    local_rank: int = field(default_factory=lambda: _parse_int_env("LOCAL_RANK", 0))
    world_size: int = field(default_factory=lambda: _parse_int_env("WORLD_SIZE", 1))
    enable_dp_move_up: bool = os.getenv("DP_MOVE_UP_ENABLE", "0") == "1"
    enable_dp_partition_up: bool = os.getenv("DP_PARTITION_UP_ENABLE", "0") == "1"

    # 模型框架类型， ATB 或者 MS, 默认为ATB
    framework_backend: str = field(default_factory=lambda: os.getenv('MINDIE_LLM_FRAMEWORK_BACKEND', "ATB").lower())


    performance_prefix_tree: bool = field(default_factory=get_performance_prefix_tree)

    async_inference: bool = field(default_factory=lambda: os.getenv("MINDIE_ASYNC_SCHEDULING_ENABLE", "0") == "1")

    def __post_init__(self):
        # 校验
        if self.log_file_maxsize < 0 or self.log_file_maxsize > 524288000:
            raise ValueError("MINDIE_LLM_PYTHON_LOG_MAXSIZE should between 0 and 524288000 (500MB), "
                                f"but get {self.log_file_maxsize}.")
        if self.log_file_maxnum < 0 or self.log_file_maxnum > 64:
            raise ValueError("MINDIE_LLM_PYTHON_LOG_MAXNUM should between 0 and 64, "
                                f"but get {self.log_file_maxnum}.")

        if self.reserved_memory_gb >= 64 or self.reserved_memory_gb < 0:
            raise ValueError("RESERVED_MEMORY_GB should be in the range of 0 to 64, 64 is not inclusive.")
     
        if self.memory_fraction <= 0 or self.memory_fraction > 1.0:
            raise ValueError("NPU_MEMORY_FRACTION should be in the range of 0 to 1.0, 0.0 is not inclusive.")

        if self.world_size < 0:
            raise ValueError("WORLD_SIZE should not be a number less than 0.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("RANK should be in the range of 0 to WORLD_SIZE, WORLD_SIZE is not inclusive.")
        if self.local_rank < 0 or self.local_rank >= self.world_size:
            raise ValueError("LOCAL_RANK should be in the range of 0 to WORLD_SIZE, WORLD_SIZE is not inclusive.")
        
        if len(self.benchmark_filepath) > 1024:
            raise ValueError("The path length of MINDIE_LLM_BENCHMARK_FILEPATH exceeds the limit 1024 characters.")
        
        if not pathlib.Path(self.benchmark_filepath).is_absolute():
            raise ValueError("The path of MINDIE_LLM_BENCHMARK_FILEPATH must be absolute.")
        
        if pathlib.Path(self.benchmark_filepath).is_dir():
            raise ValueError("The path of MINDIE_LLM_BENCHMARK_FILEPATH is a director and not a file.")
        
        if pathlib.Path(self.benchmark_filepath).exists():
            if not os.access(pathlib.Path(self.benchmark_filepath), os.R_OK):
                raise PermissionError("The path of MINDIE_LLM_BENCHMARK_FILEPATH is not permitted to be read.")

        if self.framework_backend not in {"atb", "ms"}:
            raise ValueError("MINDIE_LLM_FRAMEWORK_BACKEND must be 'ATB' or 'MS'")


def _parse_int_env(var_name: str, default: int) -> int:
    try:
        value = os.getenv(var_name, str(default))
        return int(value)
    except ValueError:
        from mindie_llm.utils.log.logging import logger
        logger.error(f"Environment variable {var_name} has an invalid value. Using default value {default}.")
        return default


ENV = EnvVar()
