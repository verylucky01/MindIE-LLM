set(OPENSOURCE_COMPONENT_NAME "prometheus-cpp")
set(OPENSOURCE_COMPONENT_DIR "${THIRD_PARTY_SRC_DIR}/${OPENSOURCE_COMPONENT_NAME}")
set(PROMETHEUS_OUTPUT_DIR "${THIRD_PARTY_OUTPUT_DIR}/${OPENSOURCE_COMPONENT_NAME}")

if(EXISTS "${PROMETHEUS_OUTPUT_DIR}/include/prometheus/exposer.h")
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already exists, skipping build and install.")
    return()
endif()

if(NOT EXISTS "${OPENSOURCE_COMPONENT_DIR}/include/prometheus/exposer.h")
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} not installed, building and installing...")
    if(NOT EXISTS "${OPENSOURCE_COMPONENT_DIR}/CMakeLists.txt")
        message(STATUS "${OPENSOURCE_COMPONENT_NAME} not found, downloading and extracting...")
        download_open_source(${OPENSOURCE_COMPONENT_NAME})
    else()
        message(STATUS "${OPENSOURCE_COMPONENT_NAME} already download, skipping download and extract.")
    endif()

    set(PROMETHEUS_BUILD_DIR "${OPENSOURCE_COMPONENT_DIR}/build")
    file(MAKE_DIRECTORY ${PROMETHEUS_BUILD_DIR})
    execute_process(COMMAND sed -i "1i add_link_options(-s)" ${OPENSOURCE_COMPONENT_DIR}/CMakeLists.txt)
    execute_process(COMMAND sed -i "1i add_compile_options(-ftrapv)" ${OPENSOURCE_COMPONENT_DIR}/CMakeLists.txt)
    execute_process(COMMAND sed -i "1i add_compile_options(-D_FORTIFY_SOURCE=2 -O2)" ${OPENSOURCE_COMPONENT_DIR}/CMakeLists.txt)
    execute_process(COMMAND sed -i "1i add_compile_options(-fstack-protector-strong)" ${OPENSOURCE_COMPONENT_DIR}/CMakeLists.txt)
    execute_process(COMMAND sed -i "1i add_link_options(-Wl,-z,relro,-z,now)" ${OPENSOURCE_COMPONENT_DIR}/CMakeLists.txt)
    # configure
    execute_process(
        COMMAND cmake 
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=ON
            -DENABLE_PUSH=OFF
            -DENABLE_COMPRESSION=OFF
            -DENABLE_TESTING=OFF
            -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}"
            -DCMAKE_INSTALL_PREFIX=${PROMETHEUS_OUTPUT_DIR}
            ${OPENSOURCE_COMPONENT_DIR}
        WORKING_DIRECTORY ${PROMETHEUS_BUILD_DIR}
    )
    execute_process(
        COMMAND make clean
        WORKING_DIRECTORY ${PROMETHEUS_BUILD_DIR}
    )
    execute_process(
        COMMAND make -j${thread_num}
        WORKING_DIRECTORY ${PROMETHEUS_BUILD_DIR}
    )
    execute_process(
        COMMAND make install
        WORKING_DIRECTORY ${PROMETHEUS_BUILD_DIR}
    )
else()
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already exists, skipping build and install.")
endif()
