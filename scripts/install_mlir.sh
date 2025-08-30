LLVM_TARGETS="Native;NVPTX"
LLVM_PROJECTS="mlir;llvm;lld"
LLVM_BUILD_TYPE="RelWithDebInfo"

./build-llvm-project.sh 2>&1 | tee _mlir.build.log