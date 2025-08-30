CLANG_VERSION=${CLANG_VERSION:-21}
sudo update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-${CLANG_VERSION}   100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${CLANG_VERSION} 100
sudo update-alternatives --install /usr/bin/ld      ld      /usr/bin/ld.lld-${CLANG_VERSION}  100

# pick if you have multiple versions:
sudo update-alternatives --config clang
sudo update-alternatives --config clang++
