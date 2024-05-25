git clone https://github.com/tree-sitter/tree-sitter-python
cd tree-sitter-python || exit
git checkout 62827156d01c74dc1538266344e788da74536b8a || exit
cd ..
git clone https://github.com/tree-sitter/tree-sitter-java
cd tree-sitter-java || exit
git checkout 3c24aa9365985830421a3a7b6791b415961ea770 || exit
cd ..
git clone https://github.com/tree-sitter/tree-sitter-cpp
cd tree-sitter-cpp || exit
git checkout -f 03fa93db133d6048a77d4de154a7b17ea8b9d076
cd ..
git clone https://github.com/tree-sitter/tree-sitter-c-sharp
cd tree-sitter-c-sharp || exit
git checkout fcacbeb4af6bcdcfb4527978a997bb03f4fe086d
cd ..
