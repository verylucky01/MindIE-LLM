# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import time

import torch
from torch import nn
from tqdm.auto import tqdm

from mindie_llm.runtime.layers.fused_moe.fused_moe import FusedMoE
from mindie_llm.runtime.utils.loader.weight_utils import WeightsFileHandler
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.utils.log.logging import logger


_BAR_FORMAT = "{desc}: {l_bar}{bar}| Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


class DefaultModelLoader:
    def __init__(self):
        self._counter_before_loading_weights: float = 0.0
        self._counter_after_loading_weights: float = 0.0
        self._loaded_weight_names = []
        self._weight_file_handler = None

    def load_weights(self, model: nn.Module, model_path: str) -> None:
        self._counter_before_loading_weights = time.perf_counter()

        # Traverse module and map to corresponding weight
        self._weight_file_handler = WeightsFileHandler(model_path, ".safetensors")
        self._load_modules(model)
        self._weight_file_handler.release_file_handler()

        self._counter_after_loading_weights = time.perf_counter()
        logger.info(
            "Loading weights took %.2f seconds",
            self._counter_after_loading_weights - self._counter_before_loading_weights,
        )

    def _get_total_leaf_modules(self, module: nn.Module, prefix: str = "") -> dict[str, nn.Module]:
        if len(list(module.children())) == 0:
            return {prefix: module}
        leaf_modules_dict = {}
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            leaf_modules_dict.update(self._get_total_leaf_modules(child, child_prefix))
        return leaf_modules_dict
    
    def _load_modules_with_progress(self, modules_dict: dict, pbar: tqdm,):
        for prefix, module in modules_dict.items():
            if hasattr(module, "prefix") and isinstance(module.prefix, list):
                shard_id = 0
                for weight_prefix in module.prefix:
                    for weight_suffix, param in module.named_parameters():
                        full_param_name = f"{weight_prefix}.{weight_suffix}"
                        loaded_weight = self._weight_file_handler.get_tensor(full_param_name)

                        if param is None:
                            continue

                        param.weight_loader(param, loaded_weight, shard_id)

                    shard_id += 1
            else:
                for weight_suffix, param in module.named_parameters():
                    full_param_name = f"{prefix}.{weight_suffix}"
                    try:
                        loaded_weight = self._weight_file_handler.get_tensor(full_param_name)
                    except ValueError as e:
                        if "Weight file was not found" in str(e) and hasattr(module, "prefix"):
                            loaded_weight = self._weight_file_handler.get_tensor(f"{module.prefix}.{weight_suffix}")
                    if param is None:
                        continue

                    param.weight_loader(param, loaded_weight)

                    if isinstance(module, FusedMoE):
                        module.weight_loader(loaded_weight, full_param_name)

            # Process weights after loading weights
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizationMethodBase):
                quant_method.process_weights_after_loading(module)

            pbar.update(1)

    def _load_modules(self, model: nn.Module):
        """Traverse module and load from corresponding safetensors checkpoint.
           return not loaded module names in model
        """
        leaf_modules_dict = self._get_total_leaf_modules(model)
        
        pbar = tqdm(total=len(leaf_modules_dict),
                    desc="Loading safetensors checkpoint shards in the lazy mode",
                    disable=get_parallel_info_manager().rank,
                    bar_format=_BAR_FORMAT,
                    unit="module")
        
        self._load_modules_with_progress(leaf_modules_dict, pbar)
        pbar.close()
