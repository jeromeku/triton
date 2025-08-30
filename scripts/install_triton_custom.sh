#!/bin/bash

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
LLVM_PROJECT_PATH=${LLVM_PROJECT_PATH:-"$REPO_ROOT/llvm-project"}
LLVM_BUILD_PATH=${LLVM_BUILD_PATH:-"$LLVM_PROJECT_PATH/build"}

export LLVM_BUILD_DIR=${LLVM_BUILD_PATH}
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR

echo "Using LLVM_BUILD_DIR: ${LLVM_BUILD_DIR}"

# Optional: speed up
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_BUILD_WITH_CCACHE=true
export TRITON_REL_BUILD_WITH_ASSERTS=1
export TRITON_NO_WERROR=1
# Build Triton (editable install)
cd ${REPO_ROOT}

# make dev-install-requires
# uv pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
# uv pip uninstall triton pytorch-triton
rm -rf build
pip install -v -e . --no-build-isolation 2>&1 | tee _triton_install.txt
