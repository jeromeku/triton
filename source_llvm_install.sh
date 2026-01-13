#!/usr/bin/env bash

set -euo pipefail

# Repository root inferred from this script’s location.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PYTHON ?= python
PYTHON=${PYTHON:-python}

# BUILD_DIR := $(shell cd python; $(PYTHON) -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
BUILD_DIR="$(
  cd "${REPO_ROOT}/python" && \
  "${PYTHON}" -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'
)"

# TRITON_OPT := $(BUILD_DIR)/bin/triton-opt
TRITON_OPT="${BUILD_DIR}/bin/triton-opt"

# LLVM_BUILD_PATH ?= "$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/.llvm-project/build"
# Here we approximate dirname(realpath(Makefile)) with REPO_ROOT.
LLVM_BUILD_PATH=${LLVM_BUILD_PATH:-"${REPO_ROOT}/.llvm-project/build"}

# NUM_PROCS ?= 8
NUM_PROCS=${NUM_PROCS:-8}
echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHON=${PYTHON}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "TRITON_OPT=${TRITON_OPT}"
echo "LLVM_BUILD_PATH=${LLVM_BUILD_PATH}"
echo "NUM_PROCS=${NUM_PROCS}"

# # # Run the Makefile target and log output.
# (cd "${REPO_ROOT}" && make dev-install-llvm) 2>&1 | tee "${REPO_ROOT}/llvm_install.log"
ninja -C ${LLVM_BUILD_PATH} 2>&1 | tee llvm_build.log