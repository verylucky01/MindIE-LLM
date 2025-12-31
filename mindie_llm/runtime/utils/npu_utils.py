# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from enum import Enum
from dataclasses import dataclass
import threading
from typing import Any

import torch_npu
import torch

from mindie_llm.runtime.utils.command_executor_utils import execute_command


class Topo(str, Enum):
    """Enumerates supported inter-device communication topologies on Ascend NPUs."""
    pcie = "pcie"
    hccs = "hccs"
    xlink = "xlink"


class CommunicationLibrary(str, Enum):
    """Enumerates supported collective communication libraries for Ascend NPUs."""
    hccl = "hccl"


class AscendDeviceType(str, Enum):
    """Enumerates supported Ascend NPU device types based on SoC name."""
    ASCEND_910B = "ASCEND_910B"
    ASCEND_910_93 = "ASCEND_910_93"
    ASCEND_310P = "ASCEND_310P"
    ASCEND_910_95 = "ASCEND_910_95"


@dataclass
class PlatformInfo:
    """
    Singleton class to capture and cache platform-specific information for Ascend NPU devices.

    This class auto-detects the NPU SoC name, device type, and communication capabilities
    using system commands and torch APIs. It ensures thread-safe, lazy initialization.

    Attributes:
        soc_name (str): Name of the System-on-Chip. Currently unused, reserved.
        only_supports_nz (bool): Whether the device only supports the NZ data format.
        only_supports_bf16 (bool): Whether the device supports BF16 precision (False for NZ-only devices).
        need_nz (bool): Legacy flag for compatibility with ATB models. Should be aligned with only_supports_nz.

    Note:
        This class is implemented as a singleton using __new__ and threading.Lock to ensure
        platform info is initialized only once during the process lifetime.
    """
    soc_name: str = ""
    only_supports_nz: bool = False
    only_supports_bf16: bool = True
    need_nz: bool = False  # Legacy compatibility flag for ATB models; runtime does not use this

    # Singleton pattern fields
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> 'PlatformInfo':
        """
        Override __new__ to enforce singleton pattern.

        Ensures only one instance of PlatformInfo is created per process.
        Uses thread-safe double-checked locking pattern.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __post_init__(self) -> None:
        """
        Post-initialization hook called after dataclass __init__.

        Queries the NPU SoC name  and sets device capabilities
        based on name lists. Updates only_supports_nz, need_nz, and only_supports_bf16.
        """
        self.soc_name = torch.npu.get_device_properties().name

        # SoC names known to support only NZ format and not BF16
        nz_only_names = {
            "Ascend910PremiumA",
            "Ascend910ProA",
            "Ascend910A",
            "Ascend910ProB",
            "Ascend910B",
            "Ascend310P1",
            "Ascend310P2",
            "Ascend310P3",
            "Ascend310P4",
            "Ascend310P5",
            "Ascend310P7"
        }
        if self.soc_name in nz_only_names:
            self.only_supports_nz = True
            self.need_nz = True
            self.only_supports_bf16 = False

    @staticmethod
    def is_support_hccs() -> bool:
        """
        Determines whether the system supports HCCS or XLink topology via npu-smi command.

        Executes `npu-smi info -t topo` and checks if "hccs" or "xlink" appears in the topology legend section.

        Returns:
            bool: True if HCCS or XLink is detected; False otherwise.
        """
        npu_smi_info = execute_command(["npu-smi", "info", "-t", "topo"])
        # Find the position of "Legend" to avoid false positives in device table
        legend_index = npu_smi_info.find("Legend")
        # Check if either HCCS or XLink appears before the Legend section
        if Topo.hccs in npu_smi_info[:legend_index].lower() or \
            Topo.xlink in npu_smi_info[:legend_index].lower():
            return True
        return False

    def get_device_type(self) -> AscendDeviceType:
        """
        Determines the Ascend device type based on the detected SoC name.
        Maps numeric SoC name lists to known Ascend device types.

        Returns:
            AscendDeviceType: The corresponding device type.

        Raises:
            RuntimeError: If the SoC name is unrecognized or unsupported.
        """
        if self.soc_name in {
            "Ascend910B1",
            "Ascend910B2",
            "Ascend910B2C",
            "Ascend910B3",
            "Ascend910B4",
            "Ascend910B4_1",
        }:
            return AscendDeviceType.ASCEND_910B
        elif self.soc_name in {
            "Ascend910_9391",
            "Ascend910_9392",
            "Ascend910_9381",
            "Ascend910_9382",
            "Ascend910_9372",
            "Ascend910_9362"
        }:
            return AscendDeviceType.ASCEND_910_93
        elif self.soc_name in {
            "Ascend310P1",
            "Ascend310P2",
            "Ascend310P3",
            "Ascend310P4",
            "Ascend310P5",
            "Ascend310P7"
        }:
            return AscendDeviceType.ASCEND_310P
        else:
            raise RuntimeError(f"Can not support soc_name: {self.soc_name}.")


def get_platform_info() -> PlatformInfo:
    """
    Returns the singleton instance of PlatformInfo.

    This is the recommended entry point to access platform information.
    The instance is lazily initialized on first call.

    Returns:
        PlatformInfo: The singleton platform info instance.
    """
    return PlatformInfo()
