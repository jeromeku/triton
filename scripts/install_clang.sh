#!/bin/bash

set -euo pipefail

CLANG_VERSION=${CLANG_VERSION:-21}
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- ${CLANG_VERSION}