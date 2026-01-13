#!/usr/bin/env bash

set -euo pipefail

# pip install -r python/requirements.txt; pip install nanobind; 
# Run source_llvm_install.sh


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

TRITON_BUILD_WITH_CLANG_LLD=1
TRITON_BUILD_WITH_CCACHE=0

LLVM_INCLUDE_DIRS=$(LLVM_BUILD_PATH)/include
LLVM_LIBRARY_DIR=$(LLVM_BUILD_PATH)/lib
LLVM_SYSPATH=$(LLVM_BUILD_PATH)

DEBUG=1
REL_WITH_DEB_INFO=0
TRITON_REL_BUILD_WITH_ASSERTS=1

uv pip install -r python/test-requirements.txt
DEBUG=1 uv pip install -e . --no-build-isolation -v 2>&1 | tee triton.build.log

