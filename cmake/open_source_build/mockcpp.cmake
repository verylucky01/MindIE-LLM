set(MOCKCPP_SRC_DIR "${THIRD_PARTY_SRC_DIR}/mockcpp")
set(MOCKCPP_BUILD_DIR "${THIRD_PARTY_SRC_DIR}/mockcpp/build")
set(MOCKCPP_OUTPUT_DIR "${THIRD_PARTY_OUTPUT_DIR}/mockcpp")

if(EXISTS "${MOCKCPP_OUTPUT_DIR}/lib" AND EXISTS "${MOCKCPP_OUTPUT_DIR}/include")
    message(STATUS "mockcpp already installed, skipping build and install.")
    return()
else()
    message(STATUS "mockcpp not found, downloading and extracting...")

    if(NOT EXISTS ${MOCKCPP_SRC_DIR})
        download_open_source(mockcpp)
    endif()

    file(MAKE_DIRECTORY ${MOCKCPP_BUILD_DIR})
    set(CXX_FLAGS "-fPIC -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
    execute_process(
        COMMAND bash -c "patch -p1 -f < ${PROJECT_SOURCE_DIR}/tests/update.patch || true"
        WORKING_DIRECTORY "${MOCKCPP_SRC_DIR}"
    )

    execute_process(
        COMMAND cmake ${MOCKCPP_SRC_DIR} -DCMAKE_INSTALL_PREFIX=${MOCKCPP_OUTPUT_DIR} -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
        WORKING_DIRECTORY "${MOCKCPP_BUILD_DIR}"
    )
    execute_process(
        COMMAND cmake --build . -j${thread_num}
        WORKING_DIRECTORY "${MOCKCPP_BUILD_DIR}"
    )
    execute_process(
        COMMAND cmake --install .
        WORKING_DIRECTORY "${MOCKCPP_BUILD_DIR}"
    )
    message(STATUS "mockcpp is successfully installed to ${MOCKCPP_OUTPUT_DIR}.")
endif()
