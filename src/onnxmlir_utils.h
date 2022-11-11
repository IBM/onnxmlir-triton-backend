#ifndef ONNX_MLIR_UTILS_H
#define ONNX_MLIR_UTILS_H

#include "triton/core/tritonbackend.h"
#include <OnnxMlirRuntime.h>

namespace triton { namespace backend { namespace onnxmlir {

extern "C" {
    
OM_DATA_TYPE TritonDataTypeToOmDataType(TRITONSERVER_DataType datatype);

} // extern "C"

}}}  // namespace triton::backend::onnxmlir

#endif //ONNX_MLIR_UTILS_H