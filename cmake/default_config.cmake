option(ONLY_BUILD_OPENSOURCE "Just download and compile opensource, the target is in 'build/third_party'" OFF)
option(DOMAIN_LAYERED_TEST "Enable DLT test case'" OFF)
option(USE_PYTHON_TEST "Enable Python test case'" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if(DEFINED ENV{thread_num})
    set(thread_num $ENV{thread_num})
else()
    set(thread_num ${CMAKE_BUILD_PARALLEL_LEVEL})
endif()

set (DEPENDENCY_JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/dependency.json)
set (THIRD_PARTY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set (THIRD_PARTY_CACHE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.cache/downloadCache)
set (THIRD_PARTY_OUTPUT_DIR ${THIRD_PARTY_SRC_DIR}/output)