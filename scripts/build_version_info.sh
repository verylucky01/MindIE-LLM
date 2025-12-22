#!/bin/bash
function fn_build_version_info()
{
    if [ ! -f "$CODE_ROOT"/../CI/config/version.ini ]; then
        echo "version.ini is not existed,user default Version-config!"
        PACKAGE_NAME='1.0.RC3'
        MINDIEVERSION='1.0.RC3'
    else
        PACKAGE_NAME=$(cat $VERSION_INFO_FILE | grep "PackageName" | cut -d "=" -f 2)
        MINDIEVERSION=$(cat $VERSION_INFO_FILE | grep "MindIEVersion" | cut -d " " -f 2)
    fi
    echo "PackageName: $PACKAGE_NAME"
    echo "MindIEVersion: $MINDIEVERSION"
    if [[ ! "${PACKAGE_NAME}" =~ ^[1-9]\.[0-9]\. ]]; then
        echo "VERSION is invalid"
        exit 1
    fi
    
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
    MindIE-LLM : ${PACKAGE_NAME}
    MindIE-LLM Version : ${MINDIEVERSION}
    Platform : ${ARCH}
    branch : ${branch}
    commit id : ${commit_id}
EOF
}

function get_version()
{
    # 黄区
    VERSION="1.0.0"
    if [ ! -f "${CODE_ROOT}"/../CI/config/version.ini ]; then
        echo "version.ini is not existed, failed to get version info"
    else
        VERSION=$(cat ${CODE_ROOT}/../CI/config/version.ini | grep "PackageName" | cut -d "=" -f 2)
        echo $VERSION
    fi
    export MINDIE_VERSION=$VERSION
}