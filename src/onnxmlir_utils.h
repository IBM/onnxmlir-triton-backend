#ifndef ONNX_MLIR_UTILS_H
#define ONNX_MLIR_UTILS_H

#include "triton/core/tritonbackend.h"
#include <OnnxMlirRuntime.h>

namespace triton { namespace backend { namespace onnxmlir {

OM_DATA_TYPE TritonDataTypeToOmDataType(TRITONSERVER_DataType datatype);

}}}  // namespace triton::backend::onnxmlir

#endif //ONNX_MLIR_UTILS_H
