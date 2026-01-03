# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List

from transformers.configuration_utils import PretrainedConfig

from mindie_llm.runtime.utils.parameter_validators import (
    IntParameterValidator, FloatParameterValidator, RangeParamaterValidator, Field
)


@dataclass
class GenerationConfig:
    """A base class used to store model generation configuration.

    Attributes:
        pad_token_id: ID of the padding token.
        eos_token_id: ID(s) of the end-of-sequence token(s), can be an int, a list of ints, or a list of lists of ints.
    """
    pad_token_id: Optional[int] = None
    eos_token_id: Union[int, List[Union[int, List[int]]]] = None


@dataclass
class RopeScaling:
    """A base class containing the scaling configuration for the RoPE embeddings.

    Attributes:
        factor:  In most scaling types, a factor of x will enable the model to handle
            sequences of length x original maximum pre-trained length.
        rope_type: The sub-variant of RoPE to use.
        original_max_position_embeddings: The original max position embeddings used during pretraining.
        beta_fast: Only Used with `type` equals to `yarn`.
            Parameter to set the boundary for extrapolation (only) in the linear ramp function.
        beta_slow: Only Used with `type` equals to `yarn`.
            Parameter to set the boundary for extrapolation (only) in the linear ramp function
    """
    factor: float = 1.0
    rope_type: str = 'linear'

    original_max_position_embeddings: Optional[Any] = None
    beta_fast: Optional[int] = 32
    beta_slow: Optional[int] = 1


@dataclass
class HuggingFaceConfig(PretrainedConfig):
    """A base class used to store model configuration information.

    This class defines some common fields that will be called by higher-level components. Each model should implement a
    subclass that inherits from this class to initialize its specific configuration. This approach ensures that the
    fields called by higher-level components are guaranteed to exist in the configuration object's attributes, thereby
    preventing exceptions during calls. Additionally, this base class performs security checks on common parameters,
    eliminating the need for each model to implement its own security validation functions.
    Note: Whether using the base class or subclass, the `from_dict` method should be used to construct the object.
    Otherwise, parameter validation will be bypassed, potentially leaving security risks caused by invalid parameters.
    The `from_pretrained` method in this class is only used to perform security validation on the model_path passed in
    case the method is called unexpectedly. This helps prevent security risks caused by tampered weight files. Under no
    circumstances should `from_pretrained` be considered the primary method to construct a configuration object.

    Attributes:
        rope_scaling: A dict or an object of `RopeScaling` class of rope scaling configuration information, detailed in
            the `RopeScaling` class. The default value is None.
    """
    rope_scaling: RopeScaling | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parse_obj()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> 'HuggingFaceConfig':
        """Method used to construct an object of this class.

        This method is used to construct an object from a dictionary, which allows additional parameters to be passed
        via kwargs.

        Args:
            config_dict: A dictionary containing configuration parameters.
            **kwargs: Additional configuration parameters which override the configuration parameters in `config_dict`.
        """
        config_dict.update(kwargs)
        config = cls(**config_dict)
        config.validate()
        return config

    def validate(self):
        self._validate_config()
        self._validate_rope_scaling()
    
    def _parse_obj(self):
        # rope scaling
        if self.rope_scaling is None:
            self.rope_scaling = {}
        self.rope_scaling = RopeScaling(**self.rope_scaling)
        

    def _validate_config(self):
        validators = {
            "max_position_embeddings": IntParameterValidator(Field(gt=0)),
            "vocab_size": IntParameterValidator(Field(gt=0)),
        }
        for key, validator in validators.items():
            value = getattr(self, key)
            validator.validate(value, key)

    def _validate_rope_scaling(self):
        if self.rope_scaling is None:
            return

        validators = {
            "factor": FloatParameterValidator(Field(ge=-65504, le=65504)),
            "rope_type": RangeParamaterValidator(['linear', 'yarn']),
            "original_max_position_embeddings": IntParameterValidator(Field(ge=1, le=2147483647), allow_none=True),
            "beta_fast": IntParameterValidator(Field(ge=1, le=2147483647), allow_none=True),
            "beta_slow": IntParameterValidator(Field(ge=1, le=2147483647), allow_none=True),
        }

        for key, validator in validators.items():
            value = getattr(self.rope_scaling, key)
            validator.validate(value, key)
