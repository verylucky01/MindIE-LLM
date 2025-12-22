set(SOURCE_DIR ${THIRD_PARTY_SRC_DIR}/grpc/)
set(BUILD_DIR ${SOURCE_DIR}/build/)
set(INSTALL_DIR ${THIRD_PARTY_OUTPUT_DIR}/grpc)
set(CHECK_FILE_DAGGER ${INSTALL_DIR}/bin/acountry)
set(GRPC_SOURCE_DIR "${THIRD_PARTY_SRC_DIR}/grpc/grpc")
set(PROTOBUF_SOURCE_DIR "${THIRD_PARTY_SRC_DIR}/grpc/protobuf")
set(ABSEIL_SOURCE_DIR "${THIRD_PARTY_SRC_DIR}/grpc/abseil-cpp")
set(CARES_SOURCE_DIR "${THIRD_PARTY_SRC_DIR}/grpc/cares")
set(RE2_SOURCE_DIR "${THIRD_PARTY_SRC_DIR}/grpc/re2")
set(ZLIB_SOURCE_DIR "${THIRD_PARTY_SRC_DIR}/grpc/zlib")


if(EXISTS ${CHECK_FILE_DAGGER})
    message("-- grpc: ${CHECK_FILE_DAGGER}")
    message("-- grpc: has been built, ignored")
    return()
endif()

if(NOT EXISTS "${THIRD_PARTY_SRC_DIR}/grpc")  # 修改为组件自己的判断逻辑
    message(STATUS "grpc not found, downloading and extracting...")

    set(OPENSOURCE_COMPONENT_NAME "grpc")
    download_open_source_with_path(${OPENSOURCE_COMPONENT_NAME} ${GRPC_SOURCE_DIR})

    set(OPENSOURCE_COMPONENT_NAME "protobuf")
    download_open_source_with_path(${OPENSOURCE_COMPONENT_NAME} ${PROTOBUF_SOURCE_DIR})

    set(OPENSOURCE_COMPONENT_NAME "abseil-cpp")
    download_open_source_with_path(${OPENSOURCE_COMPONENT_NAME} ${ABSEIL_SOURCE_DIR})

    set(OPENSOURCE_COMPONENT_NAME "cares")
    download_open_source_with_path(${OPENSOURCE_COMPONENT_NAME} ${CARES_SOURCE_DIR})

    set(OPENSOURCE_COMPONENT_NAME "re2")
    download_open_source_with_path(${OPENSOURCE_COMPONENT_NAME} ${RE2_SOURCE_DIR})

    set(OPENSOURCE_COMPONENT_NAME "zlib")
    download_open_source_with_path(${OPENSOURCE_COMPONENT_NAME} ${ZLIB_SOURCE_DIR})
endif()

function(build_grpc USE_CXX11_ABI)
    message("---------------> grpc: ${CHECK_FILE_DAGGER}")
    # unzip and Patch gRPC 3rdparty
    # grpc
    set(GPRC_DIR ${THIRD_PARTY_SRC_DIR}/grpc)
    exec_program(mkdir ${GRPC_SOURCE_DIR} ARGS -p ${GRPC_SOURCE_DIR}/SOURCE ${GPRC_DIR})
    exec_program(tar ${GRPC_SOURCE_DIR} ARGS -xf ${GRPC_SOURCE_DIR}/*.tar.gz -C ${GRPC_SOURCE_DIR}/SOURCE)
    exec_program(ls ${GRPC_SOURCE_DIR} ARGS ${GRPC_SOURCE_DIR}/SOURCE | head -n 1 OUTPUT_VARIABLE GRPC_FILE_NAME)
    execute_process(COMMAND /bin/sh -c "cd ${GRPC_SOURCE_DIR}/SOURCE/${GRPC_FILE_NAME} && grep 'Patch' ${GRPC_SOURCE_DIR}/*.spec |awk '{print $2}' | while read -r patch ; do if [ -e ${GRPC_SOURCE_DIR}/$patch ] ;then patch -p1 < ${GRPC_SOURCE_DIR}/$patch ; fi ; done")
    exec_program(cp ${GRPC_SOURCE_DIR} ARGS -r ${GRPC_SOURCE_DIR}/SOURCE/${GRPC_FILE_NAME}/* ${GPRC_DIR})
    # exec_program(patch ${GPRC_DIR} ARGS ${GPRC_DIR}CMakeLists.txt ${PROJECT_3RDPARTY_SRC_DIR}/grpc/grpc-safe-compile.patch)
    ## mkdir xds envoy-api googleapis opencensus-proto/src 
    exec_program(mkdir ${GPRC_DIR} ARGS -p ${GPRC_DIR}/third_party/xds)
    exec_program(mkdir ${GPRC_DIR} ARGS -p ${GPRC_DIR}/third_party/envoy-api)
    exec_program(mkdir ${GPRC_DIR} ARGS -p ${GPRC_DIR}/third_party/googleapis)
    exec_program(mkdir ${GPRC_DIR} ARGS -p ${GPRC_DIR}/third_party/opencensus-proto/src)
    # ZLIB
    set(GRPC_ZLIB_DIR ${THIRD_PARTY_SRC_DIR}/grpc/third_party/zlib)
    exec_program(mkdir ${ZLIB_SOURCE_DIR} ARGS -p ${ZLIB_SOURCE_DIR}/SOURCE ${GRPC_ZLIB_DIR})
    exec_program(tar ${ZLIB_SOURCE_DIR} ARGS -xf ${ZLIB_SOURCE_DIR}/*.tar.xz -C ${ZLIB_SOURCE_DIR}/SOURCE)
    exec_program(ls ${ZLIB_SOURCE_DIR} ARGS ${ZLIB_SOURCE_DIR}/SOURCE | head -n 1 OUTPUT_VARIABLE ZLIB_FILE_NAME)
    execute_process(COMMAND /bin/sh -c "cd ${ZLIB_SOURCE_DIR}/SOURCE/${ZLIB_FILE_NAME} && grep 'Patch' ${ZLIB_SOURCE_DIR}/*.spec |awk '{print $2}' | while read -r patch ; do if [ -e ${ZLIB_SOURCE_DIR}/$patch ] ;then patch -p1 < ${ZLIB_SOURCE_DIR}/$patch ; fi ; done")
    exec_program(cp ${ZLIB_SOURCE_DIR} ARGS -r ${ZLIB_SOURCE_DIR}/SOURCE/${ZLIB_FILE_NAME}/* ${GRPC_ZLIB_DIR})
    # PROTOBUF
    message("start patch PROTOBUF")
    set(GRPC_PROTOBUF_DIR ${THIRD_PARTY_SRC_DIR}/grpc/third_party/protobuf)
    exec_program(mkdir ${PROTOBUF_SOURCE_DIR} ARGS -p ${PROTOBUF_SOURCE_DIR}/SOURCE ${GRPC_PROTOBUF_DIR})
    exec_program(tar ${PROTOBUF_SOURCE_DIR} ARGS -xzf ${PROTOBUF_SOURCE_DIR}/*.tar.gz -C ${PROTOBUF_SOURCE_DIR}/SOURCE)
    exec_program(ls ${PROTOBUF_SOURCE_DIR} ARGS ${PROTOBUF_SOURCE_DIR}/SOURCE | head -n 1 OUTPUT_VARIABLE PROTOBUF_FILE_NAME)
    execute_process(COMMAND /bin/sh -c "cd ${PROTOBUF_SOURCE_DIR}/SOURCE/${PROTOBUF_FILE_NAME} && grep 'Patch' ${PROTOBUF_SOURCE_DIR}/*.spec |awk '{print $2}' | while read -r patch ; do if [ -e ${PROTOBUF_SOURCE_DIR}/$patch ] ;then patch -p1 < ${PROTOBUF_SOURCE_DIR}/$patch ; fi ; done")
    exec_program(cp ${PROTOBUF_SOURCE_DIR} ARGS -r ${PROTOBUF_SOURCE_DIR}/SOURCE/${PROTOBUF_FILE_NAME}/* ${GRPC_PROTOBUF_DIR})
    # ABSEIL
    message("start patch ABSEIL")
    set(GRPC_ABSEIL_DIR ${THIRD_PARTY_SRC_DIR}/grpc/third_party/abseil-cpp)
    exec_program(mkdir ${ABSEIL_SOURCE_DIR} ARGS -p ${ABSEIL_SOURCE_DIR}/SOURCE ${GRPC_ABSEIL_DIR})
    exec_program(tar ${ABSEIL_SOURCE_DIR} ARGS -xzf ${ABSEIL_SOURCE_DIR}/*.tar.gz -C ${ABSEIL_SOURCE_DIR}/SOURCE)
    exec_program(ls ${ABSEIL_SOURCE_DIR} ARGS ${ABSEIL_SOURCE_DIR}/SOURCE | head -n 1 OUTPUT_VARIABLE ABSEIL_FILE_NAME)
    # exec_program(patch ${ABSEIL_SOURCE_DIR} ARGS ${ABSEIL_SOURCE_DIR}/SOURCE/${ABSEIL_FILE_NAME}/CMakeLists.txt ${PROJECT_3RDPARTY_SRC_DIR}/grpc/absl_safe_compile.patch)
    execute_process(COMMAND /bin/sh -c "cd ${ABSEIL_SOURCE_DIR}/SOURCE/${ABSEIL_FILE_NAME} && grep 'Patch' ${ABSEIL_SOURCE_DIR}/*.spec |awk '{print $2}' | while read -r patch ; do if [ -e ${ABSEIL_SOURCE_DIR}/$patch ] ;then patch -p1 < ${ABSEIL_SOURCE_DIR}/$patch ; fi ; done")
    exec_program(cp ${ABSEIL_SOURCE_DIR} ARGS -r ${ABSEIL_SOURCE_DIR}/SOURCE/${ABSEIL_FILE_NAME}/* ${GRPC_ABSEIL_DIR})

    message("-------------------->before fix options.h")
    file(READ "${GRPC_ABSEIL_DIR}/absl/base/options.h" ABSL_INTERNAL_OPTIONS_H_CONTENTS)
    string(REGEX REPLACE
        "#define ABSL_OPTION_USE_STD_([^ ]*) 2"
        "#define ABSL_OPTION_USE_STD_\\1 0"
        ABSL_INTERNAL_OPTIONS_H_PINNED
        "${ABSL_INTERNAL_OPTIONS_H_CONTENTS}")
    file(WRITE "${GRPC_ABSEIL_DIR}/absl/base/options.h" "${ABSL_INTERNAL_OPTIONS_H_PINNED}")
    message("-------------------->after fix options.h")

    # RE2
    message("start patch RE2")
    set(GRPC_RE2_DIR ${THIRD_PARTY_SRC_DIR}/grpc/third_party/re2)
    exec_program(mkdir ${RE2_SOURCE_DIR} ARGS -p ${RE2_SOURCE_DIR}/SOURCE ${GRPC_RE2_DIR})
    exec_program(tar ${RE2_SOURCE_DIR} ARGS -xzf ${RE2_SOURCE_DIR}/*.tar.gz -C ${RE2_SOURCE_DIR}/SOURCE)
    exec_program(ls ${RE2_SOURCE_DIR} ARGS ${RE2_SOURCE_DIR}/SOURCE | head -n 1 OUTPUT_VARIABLE RE2_FILE_NAME)
    execute_process(COMMAND /bin/sh -c "cd ${RE2_SOURCE_DIR}/SOURCE/${RE2_FILE_NAME} && grep 'Patch' ${RE2_SOURCE_DIR}/*.spec |awk '{print $2}' | while read -r patch ; do if [ -e ${RE2_SOURCE_DIR}/$patch ] ;then patch -p1 < ${RE2_SOURCE_DIR}/$patch ; fi ; done")
    exec_program(cp ${RE2_SOURCE_DIR} ARGS -r ${RE2_SOURCE_DIR}/SOURCE/${RE2_FILE_NAME}/* ${GRPC_RE2_DIR})
    # CARES
    message("start patch CARES")
    set(GRPC_CARES_DIR ${THIRD_PARTY_SRC_DIR}/grpc/third_party/cares/cares)
    exec_program(mkdir ${CARES_SOURCE_DIR} ARGS -p ${CARES_SOURCE_DIR}/SOURCE ${GRPC_CARES_DIR})
    exec_program(tar ${CARES_SOURCE_DIR} ARGS -xzf ${CARES_SOURCE_DIR}/*.tar.gz -C ${CARES_SOURCE_DIR}/SOURCE)
    exec_program(ls ${CARES_SOURCE_DIR} ARGS ${CARES_SOURCE_DIR}/SOURCE | head -n 1 OUTPUT_VARIABLE CARES_FILE_NAME)
    execute_process(COMMAND /bin/sh -c "cd ${CARES_SOURCE_DIR}/SOURCE/${CARES_FILE_NAME} && grep 'Patch' ${CARES_SOURCE_DIR}/*.spec |awk '{print $2}' | while read -r patch ; do if [ -e ${CARES_SOURCE_DIR}/$patch ] ;then patch -p1 < ${CARES_SOURCE_DIR}/$patch ; fi ; done")
    exec_program(cp ${CARES_SOURCE_DIR} ARGS -r ${CARES_SOURCE_DIR}/SOURCE/${CARES_FILE_NAME}/* ${GRPC_CARES_DIR})

    # Compile gRPC
    message("---------------> star compile grpc")
    exec_program(mkdir ${BUILD_DIR} ARGS -p ${BUILD_DIR})
    exec_program(mkdir ${BUILD_DIR} ARGS -p ${INSTALL_DIR})

    EXECUTE_PROCESS(COMMAND arch COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
    # add compile options
    if (ARCHITECTURE STREQUAL "aarch64")
        set(CFLAGS_VALUE "-march=armv8-a+crc")
    endif ()
    exec_program(cmake ${BUILD_DIR} ARGS -E env CFLAGS="${CFLAGS_VALUE} -s" cmake -DgRPC_BUILD_TESTS=OFF -DgRPC_SSL_PROVIDER:STRING=package
            -DOPENSSL_ROOT_DIR=${THIRD_PARTY_OUTPUT_DIR}/openssl
            -DBUILD_SHARED_LIBS=ON
            -DCMAKE_SKIP_INSTALL_RPATH=TRUE
            -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI} -fstack-protector-all -ftrapv -s -D_FORTIFY_SOURCE=2 -O2"
            -Dprotobuf_BUILD_SHARED_LIBS=ON
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_EXAMPLES=OFF
            -DgRPC_INSTALL=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ${SOURCE_DIR})
    exec_program(make ${BUILD_DIR} ARGS clean)
    exec_program(make ${BUILD_DIR} ARGS -j${thread_num})
    exec_program(make ${BUILD_DIR} ARGS install)
    
    file(MAKE_DIRECTORY ${INSTALL_DIR}/include/src)
    file(COPY "${SOURCE_DIR}/src/core" DESTINATION "${INSTALL_DIR}/include/src")
endfunction()
build_grpc(${USE_CXX11_ABI})
