#!/bin/bash
cd $CODE_ROOT/src/kernels
bash build.sh
cp dist/mie_ops*.whl $OUTPUT_DIR
cd -