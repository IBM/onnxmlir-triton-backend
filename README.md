# onnxmlir-triton-backend

A triton backend which allows the usage of onnx-mlir compiled models (model.so) 
with the triton inference server.

At the moment there is no GPU support.

## Model Repository

The backend expects the compiled model to be named `model.so`
The complete structure looks

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      <version>/
        model.so
      <version>/
        model.so
      ...
```

## Model Configuration

Specify the backend name `onnxmlir` in the config.pbtxt:

```
backend: "onnxmlir"
```

For more options see 
[Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md).

## Build and Install

You can either build the backend and copy the shared library manually to your triton installation
or let `make install` directly install it tor your triton installation.

### Manual Install
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
```
produces `build/install/backends/onnxmlir/libtriton_onnxmlir.so`

The `libtriton_onnxmlir.so` needs to be copied to `backends/onnxmlir` directory 
in the triton installation directory (usually `/opt/tritonserver`).

### Direct Install

You can specify the triton install directory on the cmake command
 so `make install` will install it directly in your trition installation.
If triton is installed to `/opt/tritonserver` you can use

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver ..
make install
```

