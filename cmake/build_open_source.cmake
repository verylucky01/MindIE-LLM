file(READ ${DEPENDENCY_JSON_FILE} DEP_JSON_STRING)
set (DEP_JSON_STRING ${DEP_JSON_STRING})

if(NOT EXISTS ${THIRD_PARTY_CACHE_DIR})
    file(MAKE_DIRECTORY ${THIRD_PARTY_CACHE_DIR})
endif()
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/open_source_build;${CMAKE_MODULE_PATH}")

include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)
# 此处可以根据不同的场景，添加不同的三方依赖
include(spdlog)
include(nlohmannJson)
include(makeself)
include(boost)
include(openssl)
include(cpphttp)
include(prometheus)
include(grpc)
include(gloo)
include(libboundscheck)

if(DEFINED ENV{BUILD_ZONE} AND "$ENV{BUILD_ZONE}" STREQUAL "yellow")
    return()
endif()

include(hseceasy)

if(USE_PYTHONTEST_TEST OR USE_FUZZ_TEST OR DOMAIN_LAYERED_TEST)
    include(googletest)
    include(mockcpp)
endif()