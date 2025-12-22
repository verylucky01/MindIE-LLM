set(MAKESELF_SRC_DIR "${THIRD_PARTY_SRC_DIR}/makeself")
set(MAKESELF_DIR "${THIRD_PARTY_OUTPUT_DIR}/makeself")

if(EXISTS "${MAKESELF_DIR}/makeself.sh")
    message(STATUS "makeself already exists, skipping download and extract.")
    return()
endif()

if(NOT EXISTS "${MAKESELF_SRC_DIR}/makeself.sh")
    message(STATUS "makeself not found, downloading and extracting...")
    if(EXISTS "${MAKESELF_SRC_DIR}")
        file(REMOVE_RECURSE "${MAKESELF_SRC_DIR}")
    endif()
    download_open_source(makeself)
else()
    message(STATUS "makeself already exists, skipping download and extract.")
endif()

file(COPY "${MAKESELF_SRC_DIR}/" DESTINATION "${MAKESELF_DIR}")