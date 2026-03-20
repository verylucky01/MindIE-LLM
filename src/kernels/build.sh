#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

if [ $# -eq 1 ]; then
    ops=$1
else
    ops=$(python3 -c "import torch; import torch_npu; soc = torch_npu._C._npu_get_soc_version(); ops = 'ascend910b' if soc <= 250 else 'ascend910_93'; print(ops)")
fi

BASE_DIR=$(realpath "$(dirname "$0")")

if ! python3 -c "import psutil" 2>/dev/null; then
    echo "正在安装 psutil..."
    pip3 install psutil
fi
export CC=$(which gcc)
export CXX=$(which g++)

$CC --version
$CXX --version

cd $BASE_DIR/mie_ops
echo "ascendc build ops: $ops"
bash csrc/build_aclnn.sh $BASE_DIR/mie_ops $ops
./csrc/build/cann-ops-transformer-custom*.run --quiet -- --install-path=$BASE_DIR/mie_ops/opp

# 编译wheel包
cd -
python3 $BASE_DIR/setup.py build bdist_wheel
