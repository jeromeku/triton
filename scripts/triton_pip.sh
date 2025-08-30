triton-pip-install () {
  REPO_BASE_DIR=$(git rev-parse --show-toplevel)
  TRITON_BUILD_WITH_CCACHE=true TRITON_BUILD_WITH_CLANG_LLD=true \
    pip install --no-build-isolation ${REPO_BASE_DIR}
}