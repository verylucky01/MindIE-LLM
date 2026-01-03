# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import patch, MagicMock

from mindie_llm.runtime.utils.npu_utils import (
    Topo,
    CommunicationLibrary,
    PlatformInfo
)


class TestTopo(unittest.TestCase):
    """Test Topo enum values for PCIe and HCCS topologies."""
    def test_topo_enum(self):
        """Verify Topo enum strings match expected values."""
        self.assertEqual(Topo.pcie.value, "pcie")
        self.assertEqual(Topo.hccs.value, "hccs")


class TestCommunicationLibrary(unittest.TestCase):
    """Test CommunicationLibrary enum for HCCL support."""
    def test_communication_library_enum(self):
        """Check HCCL communication library value correctness."""
        self.assertEqual(CommunicationLibrary.hccl.value, "hccl")


class TestPlatformInfo(unittest.TestCase):
    """Test PlatformInfo hardware detection logic."""
    @patch('mindie_llm.runtime.utils.npu_utils.torch.npu.get_device_properties')
    def test_post_init_with_need_nz_soc(self, mock_get_soc):
        """Test SOC version that requires NZ."""
        mock_property = MagicMock()
        mock_property.name = "Ascend910PremiumA"
        mock_get_soc.return_value = mock_property
        soc_info = PlatformInfo()
        self.assertEqual(soc_info.soc_name, "Ascend910PremiumA")
        self.assertTrue(soc_info.need_nz)
        self.assertTrue(soc_info.only_supports_nz)
        self.assertFalse(soc_info.only_supports_bf16)

    @patch("mindie_llm.runtime.utils.npu_utils.torch.npu.get_device_properties")
    @patch("mindie_llm.runtime.utils.npu_utils.execute_command")
    def test_is_support_hccs_true(self, mock_execute, _):
        """Verify HCCS support detection when command returns 'hccs'."""
        mock_execute.return_value = "some info hccs Legend others"
        self.assertTrue(PlatformInfo.is_support_hccs())

    @patch("mindie_llm.runtime.utils.npu_utils.torch.npu.get_device_properties")
    @patch("mindie_llm.runtime.utils.npu_utils.execute_command")
    def test_is_support_hccs_false(self, mock_execute, _):
        """Verify no HCCS support when command returns 'pcie'."""
        mock_execute.return_value = "some info pcie Legend others"
        self.assertFalse(PlatformInfo.is_support_hccs())
