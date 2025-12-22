set(GOOGLETEST_SRC_DIR "${THIRD_PARTY_SRC_DIR}/googletest")
set(GOOGLETEST_BUILD_DIR "${GOOGLETEST_SRC_DIR}/build")
set(GOOGLETEST_OUTPUT_DIR "${THIRD_PARTY_OUTPUT_DIR}/googletest")

if(EXISTS "${GOOGLETEST_OUTPUT_DIR}/lib" AND EXISTS "${GOOGLETEST_OUTPUT_DIR}/include")
    message(STATUS "Googletest already installed, skipping build and install.")
    return()
endif()

if(NOT EXISTS "${GOOGLETEST_SRC_DIR}/include")
    download_open_source(googletest)
    message(STATUS "Googletest not installed, building and installing...")
endif()

file(MAKE_DIRECTORY ${GOOGLETEST_OUTPUT_DIR})
file(MAKE_DIRECTORY ${GOOGLETEST_BUILD_DIR})

set(CXX_FLAGS "-fPIC -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
execute_process(
    COMMAND cmake ${GOOGLETEST_SRC_DIR} -DCMAKE_INSTALL_PREFIX=${GOOGLETEST_OUTPUT_DIR} -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
    WORKING_DIRECTORY ${GOOGLETEST_BUILD_DIR}
)

execute_process(
    COMMAND cmake --build . -j${thread_num}
    WORKING_DIRECTORY "${GOOGLETEST_BUILD_DIR}"
)
execute_process(
    COMMAND cmake --install .
    WORKING_DIRECTORY "${GOOGLETEST_BUILD_DIR}"
)

message(STATUS "Googletest is successfully installed to ${GOOGLETEST_SRC_DIR}.")
