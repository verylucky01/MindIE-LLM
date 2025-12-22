#!/bin/bash
function fn_make_run_package()
{
    mkdir -p $OUTPUT_DIR/scripts $OUTPUT_DIR/lib $OUTPUT_DIR/lib/grpc $OUTPUT_DIR/bin $RELEASE_DIR/$ARCH \
        $OUTPUT_DIR/conf $OUTPUT_DIR/server/scripts/

    cp $CODE_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts
    cp $CODE_ROOT/src/server/conf/config.json $OUTPUT_DIR/conf
    cp -r $CODE_ROOT/src/server/scripts/* $OUTPUT_DIR/server/scripts

    protobuf_so_version="so.2308.0.0"
    grpc_so_list=(
        "libprotobuf.so.25.1.0" \
        "libprotobuf-lite.so.25.1.0" \
        "libabsl_bad_any_cast_impl.${protobuf_so_version}" \
        "libabsl_cordz_sample_token.${protobuf_so_version}" \
        "libabsl_failure_signal_handler.${protobuf_so_version}" \
        "libabsl_flags_parse.${protobuf_so_version}" \
        "libabsl_flags_usage.${protobuf_so_version}" \
        "libabsl_flags_usage_internal.${protobuf_so_version}" \
        "libabsl_log_flags.${protobuf_so_version}" \
        "libabsl_periodic_sampler.${protobuf_so_version}" \
        "libabsl_random_internal_distribution_test_util.${protobuf_so_version}" \
        "libabsl_scoped_set_env.${protobuf_so_version}" \
        "libprotoc.so.25.1.0" \
        "libgrpc++_reflection.so.1.60" \
        "libgrpc++.so.1.60" \
        "libgrpc++_alts.so.1.60" \
        "libgrpc++_error_details.so.1.60" \
        "libgrpc++_unsecure.so.1.60" \
        "libgrpc_authorization_provider.so.1.60" \
        "libgrpc_plugin_support.so.1.60" \
        "libgrpc_unsecure.so.37" \
        "libgrpcpp_channelz.so.1.60" \
        )
    for item in "${grpc_so_list[@]}"; do
        cp $THIRD_PARTY_OUTPUT_DIR/grpc/lib/${item} $OUTPUT_DIR/lib/grpc
    done

    cp $THIRD_PARTY_OUTPUT_DIR/boost/lib/libboost_system.so.1.87.0 $OUTPUT_DIR/lib
    cp $THIRD_PARTY_OUTPUT_DIR/boost/lib/libboost_thread.so.1.87.0 $OUTPUT_DIR/lib
    cp $THIRD_PARTY_OUTPUT_DIR/boost/lib/libboost_chrono.so.1.87.0 $OUTPUT_DIR/lib
    cp $THIRD_PARTY_OUTPUT_DIR/libboundscheck/lib/libboundscheck.so $OUTPUT_DIR/lib
    sed -i "s/MINDIELLMPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${PACKAGE_NAME}!" $OUTPUT_DIR/install.sh $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/install.sh $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/install.sh $OUTPUT_DIR/scripts/uninstall.sh

    sed -i "s/ATBMODELSETENV/latest\/atb_models\/set_env.sh/" $OUTPUT_DIR/set_env.sh
    sed -i 's|${mindie_llm_path}|${mindie_llm_path}/latest|g' $OUTPUT_DIR/set_env.sh

    # makeself ascend-mindie-llm.run
    TMP_VERSION=$(python3 -c 'import sys; print(sys.version_info[0], ".", sys.version_info[1])' | tr -d ' ')
    PY_MINOR_VERSION=${TMP_VERSION##*.}
    PY_VERSION="py3$PY_MINOR_VERSION"
    chmod +x $OUTPUT_DIR/*
    $THIRD_PARTY_OUTPUT_DIR/makeself/makeself.sh --header $CODE_ROOT/scripts/makeself-header.sh \
       --help-header $CODE_ROOT/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown \
        ${OUTPUT_DIR} $RELEASE_DIR/$ARCH/Ascend-mindie-llm_${PACKAGE_NAME}_${PY_VERSION}_linux-${ARCH}.run "Ascend-mindie-llm" ./install.sh

    mv $RELEASE_DIR/$ARCH $OUTPUT_DIR
    echo "Ascend-mindie-llm_${PACKAGE_NAME}_${PY_VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR"
}

function fn_make_debug_symbols_package() {
    mkdir -p "$OUTPUT_DIR/debug_symbols"
    debug_symbols_package_name="$OUTPUT_DIR/debug_symbols/Ascend-mindie-llm-debug-symbols_${PACKAGE_NAME}_${PY_VERSION}_linux-${ARCH}.tar.gz"
    cd "$CODE_ROOT"
    tar czpf $debug_symbols_package_name llm_debug_symbols
    echo "Build tar package for llm debug symbols: $debug_symbols_package_name"
    cd -
}

function fn_make_whl() {
    PACKAGE_NAME=$(echo $PACKAGE_NAME | sed -E 's/([0-9]+)\.([0-9]+)\.RC([0-9]+)\.([0-9]+)/\1.\2rc\3.post\4/')
    PACKAGE_NAME=$(echo $PACKAGE_NAME | sed -s 's!.T!.alpha!')
    echo "MindIELLMWHLVersion $PACKAGE_NAME"
    echo "make mindie-llm whl package"
    cd $CODE_ROOT
    python3 setup.py --setup_cmd="bdist_wheel" --version=${PACKAGE_NAME}
    cp dist/*.whl $OUTPUT_DIR
    rm -rf dist mindie_llm.egg-info
    cd -
    if [ "$build_type" = "release" ]; then
        cd $CODE_ROOT/tools
        cp $OUTPUT_DIR/lib/llm_manager_python.so $CODE_ROOT/tools/llm_manager_python_api_demo
        python3 setup.py --setup_cmd="bdist_wheel" --version=${PACKAGE_NAME}
        cp dist/*.whl $OUTPUT_DIR
        rm -rf dist llm_manager_python_api_demo.egg-info
        cd -
    fi
    echo "start to build mies tokenizer wheel"
    cd "$CODE_ROOT/src/server/tokenizer"
    python3 setup_tokenizer.py bdist_wheel
    cp -v dist/mies_tokenizer-*.whl $OUTPUT_DIR
    rm -rf *.egg-info dist
    cd -
}

function fn_build_for_ci()
{
    cd $OUTPUT_DIR
    mkdir -p include
    if ! [ "$build_type" = "release" ]; then
        cp -r $CODE_ROOT/mindie_llm .
    fi
    cp -r $CODE_ROOT/src/include/* ./include/
}