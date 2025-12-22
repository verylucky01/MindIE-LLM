#!/bin/bash
export thread_num=${MAX_COMPILE_CORE_NUM:-$(nproc)}
THIRD_PARTY_OUTPUT_DIR=$CODE_ROOT/third_party/output
CACHE_DIR=$CODE_ROOT/.cache
BUILD_DIR=$CODE_ROOT/build
OUTPUT_DIR=$CODE_ROOT/output
RELEASE_DIR=$CODE_ROOT/release
LOG_PATH="/var/log/mindie_log/"
LOG_NAME="mindie_llm_install.log"
ARCH="aarch64"
if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
    ARCH="x86_64"
    echo "it is system of x86_64"
elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
    echo "it is system of aarch64"
else
    echo "it is not system of aarch64 or x86_64"
fi
COMPILE_OPTIONS=""
build_type="debug"
BUILD_OPTION_LIST="master debug release help dlt unittest 3rd clean"
BUILD_CONFIGURE_LIST=("--use_cxx11_abi=0" "--use_cxx11_abi=1" "--ini=version" "--ini=version_item")
export LANG=C.UTF-8
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=$THIRD_PARTY_OUTPUT_DIR/openssl/lib:$THIRD_PARTY_OUTPUT_DIR/grpc/lib:$THIRD_PARTY_OUTPUT_DIR/boost/lib:$THIRD_PARTY_OUTPUT_DIR/prometheus-cpp/lib:$PYTORCH_INSTALL_PATH/lib:$PYTORCH_INSTALL_PATH/../torch.libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$BUILD_DIR/src/server/common:$BUILD_DIR/src/llm_manager:$BUILD_DIR/src/utils/lib:$BUILD_DIR/src/server/src/tokenizer:$BUILD_DIR/src/server/src/gmis/infer_backend_manager:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CODE_ROOT/src/utils/lib:$CODE_ROOT/src/utils/log/lib:$CODE_ROOT/src/old_infer_req_res/lib:$LD_LIBRARY_PATH