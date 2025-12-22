set(GLOO_SRC_DIR1 "${THIRD_PARTY_SRC_DIR}/gloo")
set(GLOO_SRC_DIR2 "${THIRD_PARTY_SRC_DIR}/pytorch/third_party/gloo")

if(EXISTS "${THIRD_PARTY_OUTPUT_DIR}/gloo/gloo/algorithm.h")
    message(STATUS "gloo already installed, skipping download.")
    return()
endif()

if(NOT EXISTS "${GLOO_SRC_DIR1}/gloo/algorithm.h" AND
   NOT EXISTS "${GLOO_SRC_DIR2}/gloo/algorithm.h")
    message(STATUS "gloo not found, downloading and extracting...")
    if(EXISTS "${GLOO_SRC_DIR1}")
        file(REMOVE_RECURSE "${GLOO_SRC_DIR1}")
    endif()
    download_open_source(gloo)
endif()

if(EXISTS "${GLOO_SRC_DIR2}/gloo/algorithm.h")
    file(COPY "${GLOO_SRC_DIR2}" DESTINATION "${THIRD_PARTY_OUTPUT_DIR}")
elseif(EXISTS "${GLOO_SRC_DIR1}/gloo/algorithm.h")
    file(COPY "${GLOO_SRC_DIR1}" DESTINATION "${THIRD_PARTY_OUTPUT_DIR}")
endif()
message(STATUS "Gloo content declared successfully.")
