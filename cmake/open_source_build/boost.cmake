set(OPENSOURCE_COMPONENT_NAME "boost")
set(OPENSOURCE_COMPONENT_DIR "${THIRD_PARTY_SRC_DIR}/${OPENSOURCE_COMPONENT_NAME}")
set(BOOST_OUTPUT_DIR "${THIRD_PARTY_OUTPUT_DIR}/${OPENSOURCE_COMPONENT_NAME}")

if(EXISTS "${BOOST_OUTPUT_DIR}/lib")
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already exists, skipping download and extract.")
    return()
endif()

if(NOT EXISTS "${OPENSOURCE_COMPONENT_DIR}/bootstrap.sh")  # 修改为组件自己的判断逻辑
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} not found, downloading and extracting...")
    if(EXISTS "${OPENSOURCE_COMPONENT_DIR}")
        message(STATUS "Clean the dir: ${OPENSOURCE_COMPONENT_DIR}")
        file(REMOVE_RECURSE "${OPENSOURCE_COMPONENT_DIR}")
    endif()
    download_open_source(${OPENSOURCE_COMPONENT_NAME})
else()
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already exists, skipping download and extract.")
endif()

function(fn_build_boost)
    file(MAKE_DIRECTORY ${BOOST_OUTPUT_DIR})
    
    set(WORKING_DIR ${OPENSOURCE_COMPONENT_DIR})
    execute_process(
        COMMAND chmod +x ./bootstrap.sh
        COMMAND chmod +x ./tools/build/src/engine/build.sh
        WORKING_DIRECTORY ${WORKING_DIR}
        RESULT_VARIABLE chmod_result
    )
    if(NOT chmod_result EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for Boost scripts.")
    endif()

    execute_process(
        COMMAND ./bootstrap.sh
        WORKING_DIRECTORY ${WORKING_DIR}
        RESULT_VARIABLE bootstrap_result
    )
    if(NOT bootstrap_result EQUAL 0)
        message(FATAL_ERROR "Failed to run bootstrap.sh for Boost.")
    endif()

    set(BOOST_CXX_FLAGS_STR "-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI} -fstack-protector-strong -ftrapv")
    set(BOOST_LINK_FLAGS_STR "-Wl,-z,now -s")

    execute_process(
        COMMAND ./b2 toolset=gcc
                -j${thread_num}
                --disable-icu --with-thread --with-regex --with-log
                --with-filesystem --with-date_time --with-chrono --with-system
                cxxflags=${BOOST_CXX_FLAGS_STR}
                linkflags=${BOOST_LINK_FLAGS_STR}
                link=shared
                threading=multi variant=release stage
                --prefix=${BOOST_OUTPUT_DIR} install
        WORKING_DIRECTORY ${WORKING_DIR}
        RESULT_VARIABLE b2_result
    )
    if(NOT b2_result EQUAL 0)
        message(FATAL_ERROR "Failed to build and install Boost.${b2_result}")
    endif()

    message(STATUS "Boost has been successfully built and installed to ${BOOST_OUTPUT_DIR}") 
endfunction()

fn_build_boost()