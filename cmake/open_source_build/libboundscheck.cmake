set(CHECK_FILE_BOUNDSCHECK ${THIRD_PARTY_SRC_DIR}/libboundscheck/include/securec.h)

message("--libboundscheck: check if need to download at ${CHECK_FILE_BOUNDSCHECK}")

# build the lib if not built yet
if(EXISTS ${CHECK_FILE_BOUNDSCHECK})
    message("-- libboundscheck: ${CHECK_FILE_BOUNDSCHECK}")
    message("-- libboundscheck: has been downloaded, ignored")
else()
    download_open_source(libboundscheck)
endif()

set(CHECK_SO_BOUNDSCHECK ${THIRD_PARTY_OUTPUT_DIR}/libboundscheck/lib/libboundscheck.so)

message("--libboundscheck: check if need to build at ${CHECK_SO_BOUNDSCHECK}")

if(EXISTS ${CHECK_SO_BOUNDSCHECK})
    message("-- libboundscheck: ${CHECK_SO_BOUNDSCHECK}")
    message("-- libboundscheck: has been built, ignored")
else()
    execute_process(
            COMMAND make
            WORKING_DIRECTORY ${THIRD_PARTY_SRC_DIR}/libboundscheck
    )
    file(COPY "${THIRD_PARTY_SRC_DIR}/libboundscheck" DESTINATION "${THIRD_PARTY_OUTPUT_DIR}")
endif()
