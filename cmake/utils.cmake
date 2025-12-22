include(FetchContent)
set(PLATFORM ${CMAKE_SYSTEM_PROCESSOR})

function(download_file url CACHE_FILE)
    if(EXISTS "${CACHE_FILE}")
        file(SIZE ${CACHE_FILE} file_size)
        if(file_size GREATER 0)
            message("File exists,no need to download,if compile failed, please delete this file and run again:${CACHE_FILE}")
            return()
        endif()
    endif()
    file(DOWNLOAD
        ${url}
        ${CACHE_FILE}
        TLS_VERIFY OFF
        STATUS download_status
    )
    if(NOT download_status EQUAL 0)
        file(REMOVE_RECURSE "${CACHE_FILE}")
        message(FATAL_ERROR "download file from ${url} failed")
    endif()
endfunction()

function(download_open_source_with_path opensource_name src_path)
    string(JSON DEPENDENCIES_COUNT LENGTH ${DEP_JSON_STRING} "dependencies")
    math(EXPR DEPENDENCIES_COUNT "${DEPENDENCIES_COUNT} - 1")
    foreach(INDEX RANGE ${DEPENDENCIES_COUNT})
        string(JSON DEP_NAME MEMBER ${DEP_JSON_STRING} "dependencies" ${INDEX})
        if(NOT "${DEP_NAME}" STREQUAL "${opensource_name}")
            continue()
        endif()

        if(src_path AND NOT src_path STREQUAL "")
            set(src_dir  ${src_path})
        else()
            set(src_dir ${THIRD_PARTY_SRC_DIR}/${DEP_NAME}) 
        endif()

        string(JSON DEP_TYPE GET ${DEP_JSON_STRING} "dependencies" ${DEP_NAME} "type")
        if("${DEP_TYPE}" STREQUAL "file")
            string(JSON DEP_URL GET ${DEP_JSON_STRING} "dependencies" ${DEP_NAME} "url")
            string(JSON CACHE_FILE_NAME GET ${DEP_JSON_STRING} "dependencies" ${DEP_NAME} "fileName")
            if(NOT "${DEP_URL}" STREQUAL "")
                string (CONFIGURE "${DEP_URL}" download_url)
                message(STATUS "begin to download ${DEP_NAME} from ${download_url}")
                download_file(${download_url} ${THIRD_PARTY_CACHE_DIR}/${CACHE_FILE_NAME})
                FetchContent_Declare(
                    ${DEP_NAME}
                    URL file://${THIRD_PARTY_CACHE_DIR}/${CACHE_FILE_NAME}
                    SOURCE_DIR ${src_dir}
                )
                FetchContent_Populate(${DEP_NAME})
            endif()
            continue()
        endif()

        if("${DEP_TYPE}" STREQUAL "git")
            string(JSON DEP_REPO GET ${DEP_JSON_STRING} "dependencies" ${DEP_NAME} "url")
            string(JSON DEP_TAG GET ${DEP_JSON_STRING} "dependencies" ${DEP_NAME} "tag")
            if(NOT "${DEP_REPO}" STREQUAL "")
                FetchContent_Declare(
                    ${DEP_NAME}
                    SOURCE_DIR ${src_dir}
                    GIT_REPOSITORY ${DEP_REPO}
                    GIT_TAG ${DEP_TAG}
                )
                message(STATUS "begin to download ${DEP_NAME} from ${DEP_REPO}")
                FetchContent_Populate(${DEP_NAME})
            endif()
        endif()
    endforeach()
endfunction()


function(download_open_source opensource_name)
    set(src_dir ${THIRD_PARTY_SRC_DIR}/${opensource_name}) 
    download_open_source_with_path(${opensource_name} ${src_dir})
endfunction()

function(get_ABI_option_value)
    if(USE_CXX11_ABI)
        return()
    endif()
    execute_process(
        COMMAND bash -c "python3 ${PROJECT_SOURCE_DIR}/scripts/get_cxx11_abi_flag.py -f 'torch'"
        RESULT_VARIABLE _status
        OUTPUT_VARIABLE _CXX_ABI_FLAG
        ERROR_QUIET
        )
    if(NOT _status EQUAL 0)
        message(WARNING "Failed to GET _GLIBCXX_USE_CXX11_ABI value from torch.")
        set(USE_CXX11_ABI 0 PARENT_SCOPE)
    else()
        string(STRIP ${_CXX_ABI_FLAG} _STRIP_CXX_ABI_FLAG)
        message(STATUS "GET _GLIBCXX_USE_CXX11_ABI value from torch:${_STRIP_CXX_ABI_FLAG}")
        set(USE_CXX11_ABI ${_STRIP_CXX_ABI_FLAG} PARENT_SCOPE)
    endif()
endfunction()