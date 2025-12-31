# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
from dataclasses import dataclass
from pathlib import Path

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig, GenerationConfig
from mindie_llm.runtime.config.configuration_utils import LLMConfig
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.utils.helpers.safety.file import safe_open
from mindie_llm.runtime.utils.npu_utils import PlatformInfo
from mindie_llm.utils.log.logging import logger


@dataclass
class MindIELLMConfig:
    """
    Dataclass which contains all related configuration.

    Attributes:
        model_name_or_path (str): model's name or local path
        hf_config (HuggingFaceConfig): model's huggingface config
        llm_config (LLMConfig): model's feature configurable features from server's config
        generation_config (Optional[GenerationConfig]): model's generation_config
        quant_config (Optional[QuantizationConfigBase]): model's quant config
    """
    model_name_or_path: str
    hf_config: HuggingFaceConfig
    llm_config: LLMConfig
    generation_config: GenerationConfig
    quant_config: QuantizationConfigBase | None = None

    def __post_init__(self):
        self.soc_info = PlatformInfo()
        self.quant_config = self._init_quant_config()

    def _init_quant_config(self):
        # get quant config class
        # NOTE: Since only `ms_model_slim.QuantizationConfig` is currently supported.
        # This is a straightforward implementation. A dispatch mechanism
        # will be needed if multiple quantization types are added in the future.
        quant_cls = QuantizationConfig

        config_files = []
        if Path(self.model_name_or_path).exists() and Path(self.model_name_or_path).is_dir():
            config_files = list(Path(self.model_name_or_path).glob("*.json"))
            config_files = [str(file) for file in config_files]
        quant_config_files = []
        for file_name in quant_cls.get_config_filenames():
            for actual_file_path in config_files:
                if actual_file_path.endswith(file_name):
                    quant_config_files.append(actual_file_path)
        if len(quant_config_files) == 0:
            logger.warning(f"Cannot find the config file for `QuantizationConfig`. "
                        f"Try to load weights in the floating points instead.")
            return None
        if len(quant_config_files) > 1:
            raise ValueError(
                f"Found multiple config files for `QuantizationConfig`: "
                f"{quant_config_files}"
            )

        quant_descs = {}
        with safe_open(quant_config_files[0], 'r', check_link=False) as f:
            quant_descs = json.load(f)

        return quant_cls.from_config(quant_descs)
