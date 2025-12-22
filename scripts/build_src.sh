#!/bin/bash
function fn_build_src()
{
    cd $BUILD_DIR

    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake .. $COMPILE_OPTIONS
    if [ "$USE_VERBOSE" == "ON" ];then
        VERBOSE=1 make -j$thread_num
    else
        make -j$thread_num
    fi
    make install
    cd -
}