export LLVM_BUILD=~/triton-debug/triton/llvm-project/build
export PATH=$LLVM_BUILD/bin:$PATH
export REPO_ROOT="$(git rev-parse --show-toplevel)"
# echo " --------------------------------------------------------- "

# echo "Checking llvm build..."
# $LLVM_BUILD/bin/llvm-config --version
# $LLVM_BUILD/bin/llvm-config --build-mode        # expect: RelWithDebInfo
# $LLVM_BUILD/bin/llvm-config --assertion-mode    # expect: ON or YES

echo " --------------------------------------------------------- "

echo "Checking triton mlir opt..."
export TRITON_BUILD=~/triton-debug/triton/build/cmake.linux-x86_64-cpython-3.12
export PATH=$TRITON_BUILD/bin:$PATH
printf 'builtin.module {}' | $TRITON_BUILD/bin/triton-opt -debug -pass-pipeline='builtin.module(canonicalize)' -
printf 'builtin.module {}' | $LLVM_BUILD/bin/mlir-opt -debug -pass-pipeline='builtin.module(cse)' -

echo " --------------------------------------------------------- "

echo "Checking triton build..."

readelf -S $TRITON_BUILD/bin/triton-opt | grep -i debug
readelf -S $REPO_ROOT/python/triton/_C/libtriton.so | grep -i debug

# or more detailed:
$LLVM_BUILD/bin/llvm-dwarfdump $TRITON_BUILD/bin/triton-opt | head

# grep -m1 '"command":' $TRITON_BUILD/compile_commands.json
# # expect to see both -O2 and -g (RelWithDebInfo) in C++ commands

echo " --------------------------------------------------------- "

echo "Checking triton ldd..."

ldd $TRITON_BUILD/bin/triton-opt | egrep 'MLIR|LLVM|libclang|lld'
