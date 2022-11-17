# onnx-mlir-triton-backend

A triton backend which allows the usage of onnx-mlir compiled models (model.so) with the triton inference server.

## Build
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
```
produces `build/install/backends/onnxmlir/libtriton_onnxmlir.so`
