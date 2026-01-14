#!/usr/bin/env bash

set -euo pipefail

# pip install -r python/requirements.txt; pip install nanobind; 
# uv pip install -r python/test-requirements.txt
# Run source_llvm_install.sh


# Repository root inferred from this script’s location.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LLVM_BUILD_PATH ?= "$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/.llvm-project/build"

# Here we approximate dirname(realpath(Makefile)) with REPO_ROOT.
LLVM_BUILD_PATH="${REPO_ROOT}/.llvm-project/build"

# NUM_PROCS ?= 8
NUM_PROCS=${NUM_PROCS:-8}

TRITON_BUILD_WITH_CLANG_LLD=1
TRITON_BUILD_WITH_CCACHE=0

LLVM_INCLUDE_DIRS=${LLVM_BUILD_PATH}/include
LLVM_LIBRARY_DIR=${LLVM_BUILD_PATH}/lib
LLVM_SYSPATH=${LLVM_BUILD_PATH}

DEBUG=0
REL_WITH_DEB_INFO=0
TRITON_REL_BUILD_WITH_ASSERTS=0
TRITON_DEBUG=1

TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=0 \
LLVM_BUILD_PATH="${LLVM_BUILD_PATH}" \
LLVM_INCLUDE_DIRS="${LLVM_BUILD_PATH}/include" \
LLVM_LIBRARY_DIR="${LLVM_BUILD_PATH}/lib" \
LLVM_SYSPATH="${LLVM_BUILD_PATH}" \
TRITON_DEBUG=1 DEBUG=0 REL_WITH_DEB_INFO=0 TRITON_REL_BUILD_WITH_ASSERTS=0 \
uv pip install -e . --no-build-isolation -v 2>&1 | tee triton.build.log

