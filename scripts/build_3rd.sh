#!/bin/bash
function fn_build_3rd()
{   
    if [ -d "$BUILD_DIR/_deps" ];then
        rm -rf $BUILD_DIR/_deps
    fi
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake .. $COMPILE_OPTIONS
    cd -
}