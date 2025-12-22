cmake_minimum_required(VERSION 3.19.0)
project("MindIE-LLM")
set(CMAKE_CXX_STANDARD 17)

if(USE_CXX11_ABI)
    add_definitions(-U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source build is forbidden!\n"
        "Please delete created:['CMakeFiles', 'CMakeCache.txt'] and create a new dir for build, such as: \n"
        "mkdir build & cd build\n"
        "cmake ..\n"
    )
endif()

include(${PROJECT_SOURCE_DIR}/cmake/default_config.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/compile_options.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/build_open_source.cmake)

if(ONLY_BUILD_OPENSOURCE)
    return()
endif()

execute_process(
    COMMAND sudo sh -c "echo 2 > /proc/sys/kernel/randomize_va_space"
    RESULT_VARIABLE result
)
if(NOT result EQUAL 0)
    message(WARNING "Failed to set ASLR level.")
endif()

if(USE_FUZZ_TEST)
    add_definitions(-DFUZZ_TEST)
endif()

file(GLOB PROTOBUF_SO_FILES "${THIRD_PARTY_OUTPUT_DIR}/grpc/lib/*.so")
set(Protobuf_INCLUDE_DIRS "${THIRD_PARTY_OUTPUT_DIR}/grpc/include")
set(PROTOBUF_PROTOC_EXECUTABLE "${THIRD_PARTY_OUTPUT_DIR}/grpc/bin/protoc")
set(GRPC_CPP_PLUGIN "${THIRD_PARTY_OUTPUT_DIR}/grpc/bin/grpc_cpp_plugin")

execute_process(
    COMMAND python3 -c "import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))"
    OUTPUT_VARIABLE PYTORCH_INSTALL_PATH
    ERROR_VARIABLE PYTHON_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYTORCH_EXTRA_LIB_SO "${PYTORCH_INSTALL_PATH}/../torch.libs")
string(FIND "${PYTHON_ERROR}" "ModuleNotFoundError: No module named 'torch'" TORCH_ERROR)
if(NOT TORCH_ERROR EQUAL -1)
    message(WARNING "未找到Python torch, 可以使用[pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu]下载x86_64版本的abi=0的torch")
    return()
    # download_build_torch()
endif()

set(OPENSSL_ROOT_DIR ${THIRD_PARTY_OUTPUT_DIR}/openssl)
set(gRPC_DIR ${THIRD_PARTY_OUTPUT_DIR}/grpc/lib/cmake/grpc)
set(utf8_range_DIR ${THIRD_PARTY_OUTPUT_DIR}/grpc/lib/cmake/utf8_range)
set(Protobuf_DIR ${THIRD_PARTY_OUTPUT_DIR}/grpc/lib/cmake/protobuf)
set(absl_DIR  ${THIRD_PARTY_OUTPUT_DIR}/grpc/lib/cmake/absl)
set(prometheus-cpp_DIR ${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/lib/cmake/prometheus-cpp)
set(Boost_DIR "${THIRD_PARTY_OUTPUT_DIR}/boost/lib/cmake/Boost-1.87.0")

find_package(prometheus-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread chrono)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED CONFIG) # protobuf和libtorch有先后关系
find_package(gRPC CONFIG REQUIRED)

find_package(Torch REQUIRED PATHS "${PYTORCH_INSTALL_PATH}/share/cmake/Torch")

message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
execute_process(
    COMMAND bash -c "python3 -m pybind11 --cmakedir"
    OUTPUT_VARIABLE command_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${command_output}")

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

# 项目自身的头文件目录（不使用SYSTEM标记，保持警告）
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/include/utils
    ${PROJECT_SOURCE_DIR}/src/utils/include
    ${PROJECT_SOURCE_DIR}/src/utils/log
    ${PROJECT_SOURCE_DIR}/mindie_llm/connector/cpp/
    ${PROJECT_SOURCE_DIR}/mindie_llm/text_generator/cpp/memory_bridge
    ${PROJECT_SOURCE_DIR}/mindie_llm/text_generator/cpp/sampler/cpu_logits_handler
    ${PROJECT_SOURCE_DIR}/mindie_llm/text_generator/cpp/sampler/cpu_logits_handler/log
    ${PROJECT_SOURCE_DIR}/src/include/llm_manager_v2
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/include/dataclass
    ${PROJECT_SOURCE_DIR}/src/include/config
    ${PROJECT_SOURCE_DIR}/src/llm_manager_v2/include/impl
    ${PROJECT_SOURCE_DIR}/src/llm_manager
    ${PROJECT_SOURCE_DIR}/src/config_manager
    ${PROJECT_SOURCE_DIR}/src/include/llm_manager
    ${PROJECT_SOURCE_DIR}/src/include/block_manager
    ${PROJECT_SOURCE_DIR}/src/include/scheduler
    ${PROJECT_SOURCE_DIR}/src/block_manager
    ${PROJECT_SOURCE_DIR}/src/scheduler
    ${PROJECT_SOURCE_DIR}/src/engine
    ${PROJECT_SOURCE_DIR}/src/executor
    ${PROJECT_SOURCE_DIR}/src/include/load_balance
    ${PROJECT_SOURCE_DIR}/src/server/infer_instances
    ${PROJECT_SOURCE_DIR}/src/server/common
    ${PROJECT_SOURCE_DIR}/src/include/request_response
    ${PROJECT_SOURCE_DIR}/proto
)

# 第三方库头文件使用SYSTEM标记来抑制警告
include_directories(SYSTEM
    ${THIRD_PARTY_OUTPUT_DIR}/nlohmann/include
    ${THIRD_PARTY_OUTPUT_DIR}/boost/include
    ${THIRD_PARTY_OUTPUT_DIR}/nlohmannJson/include
    ${THIRD_PARTY_OUTPUT_DIR}/spdlog/include
    ${THIRD_PARTY_OUTPUT_DIR}/googletest/include
    ${THIRD_PARTY_OUTPUT_DIR}/libboundscheck/include
    ${THIRD_PARTY_OUTPUT_DIR}/http/include
    ${Protobuf_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    $ENV{ASCEND_HOME_PATH}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/runtime/include
    ${PROJECT_SOURCE_DIR}/src/include/msServiceProfiler/include
    ${PROJECT_SOURCE_DIR}/third_party/output/gloo
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
    ${THIRD_PARTY_OUTPUT_DIR}/libboundscheck/lib)

add_subdirectory(proto)
add_subdirectory(src)

if(DEFINED ENV{ASCEND_HOME_PATH})
    message("Detected that 'CANN' is installed, starting to compile the module 'text_generator'.")
    add_subdirectory(mindie_llm/text_generator/cpp)
    add_subdirectory(mindie_llm/connector/cpp)
endif()

if(USE_PYTHONTEST_TEST OR USE_FUZZ_TEST OR DOMAIN_LAYERED_TEST)
    add_subdirectory(tests)
endif()

add_custom_target(
    custom_clean
    COMMENT "Begin to clean up the cache file" # 提示信息
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR} # 删除目录
    COMMAND ${CMAKE_MAKE_PROGRAM} clean
)