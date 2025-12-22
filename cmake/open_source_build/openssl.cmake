set(OPENSOURCE_COMPONENT_NAME "openssl")
set(OPENSSL_SOURCE_DIR "${THIRD_PARTY_SRC_DIR}/openssl")
set(OPENSSL_DEST_DIR ${THIRD_PARTY_OUTPUT_DIR}/openssl/)

if(NOT EXISTS "${OPENSSL_DEST_DIR}/include/openssl/aes.h")  # 修改为组件自己的判断逻辑
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} not found, downloading and extracting...")
    if(NOT EXISTS "${OPENSSL_SOURCE_DIR}")
        download_open_source_with_path(${OPENSOURCE_COMPONENT_NAME} ${OPENSSL_SOURCE_DIR})
    endif()
else()
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already exists, skipping download and extract.")
    return()
endif()

function(config_openssl USE_CXX11_ABI)

    set(CHECK_FILE_OPENSSL ${OPENSSL_DEST_DIR}/include/openssl/aes.h)

    message("-- openssl: check if need to build at ${CHECK_FILE_OPENSSL}")

    # build the lib if not built yet
    if(EXISTS ${CHECK_FILE_OPENSSL})
        message("-- openssl: ${CHECK_FILE_OPENSSL}")
        message("-- openssl: has been built, ignored")
    else()
        execute_process(COMMAND mkdir -p ${OPENSSL_DEST_DIR})
        execute_process(COMMAND chmod +x config WORKING_DIRECTORY ${OPENSSL_SOURCE_DIR})
        execute_process(COMMAND dos2unix config WORKING_DIRECTORY ${OPENSSL_SOURCE_DIR})
        execute_process(COMMAND dos2unix Configure WORKING_DIRECTORY ${OPENSSL_SOURCE_DIR})
        execute_process(COMMAND chmod +x Configure WORKING_DIRECTORY ${OPENSSL_SOURCE_DIR})
        set(LINK_FLAGS_STR "-Wl,-z,now -s")
        execute_process(COMMAND ./config --prefix=${OPENSSL_DEST_DIR}
                        --libdir=${OPENSSL_DEST_DIR}/lib
                        no-unit-test
                        no-tests
                        no-external-tests
                        CFLAGS="-fstack-protector-strong"
                        CXXFLAGS="-fstack-protector-strong -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}"
                        LDFLAGS=${LINK_FLAGS_STR}
                        WORKING_DIRECTORY ${OPENSSL_SOURCE_DIR})

        # execute make && make install
        execute_process(COMMAND make clean WORKING_DIRECTORY ${OPENSSL_SOURCE_DIR})
        execute_process(COMMAND make -j${thread_num} WORKING_DIRECTORY ${OPENSSL_SOURCE_DIR})
        execute_process(COMMAND make install_sw WORKING_DIRECTORY ${OPENSSL_SOURCE_DIR})
    endif()
endfunction()
config_openssl(${USE_CXX11_ABI})