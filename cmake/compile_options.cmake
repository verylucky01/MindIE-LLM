include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

get_ABI_option_value()

add_compile_options(-std=c++17 -fPIC)

execute_process(COMMAND arch COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
if (ARCHITECTURE STREQUAL "x86_64")
    message("Compiling PS lib for architecture: ${ARCHITECTURE}")
    add_compile_options(-mavx)
elseif (ARCHITECTURE STREQUAL "aarch64")
    message("Compiling PS lib for architecture: ${ARCHITECTURE}")
else ()
    message(FATAL_ERROR "The target arch is not supported: ${ARCHITECTURE}")
endif ()

# 项目自身的编译警告设置
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion-null")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpie")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-equal")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2")
set(LD_FLAGS_GLOBAL "-shared -rdynamic -ldl -Wl,-z,relro \
    -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none") # for atb

if(CMAKE_BUILD_TYPE STREQUAL "Debug") # debug mode
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g2 -ggdb")
else () # release mode
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
endif()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LD_FLAGS_GLOBAL}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} -pie")

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR "$ENV{MINDIE_ENABLE_PROF}" MATCHES "1")
    add_definitions(-DENABLE_PROF)
    set($ENV{MINDIE_ENABLE_PROF} "1")
endif ()

if(USE_PYTHON_TEST OR USE_FUZZ_TEST OR DOMAIN_LAYERED_TEST)
    enable_testing()
    add_compile_definitions(UT_ENABLED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
endif()

if(USE_FUZZ_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize-coverage=trace-pc")
endif()