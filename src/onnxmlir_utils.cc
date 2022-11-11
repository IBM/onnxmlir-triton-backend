#include "onnxmlir_utils.h"

namespace triton { namespace backend { namespace onnxmlir {

OM_DATA_TYPE TritonDataTypeToOmDataType(TRITONSERVER_DataType datatype)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL:
      return ONNX_TYPE_BOOL;
    case TRITONSERVER_TYPE_UINT8:
      return ONNX_TYPE_UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return ONNX_TYPE_UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return ONNX_TYPE_UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return ONNX_TYPE_UINT64;
    case TRITONSERVER_TYPE_INT8:
      return ONNX_TYPE_INT8;
    case TRITONSERVER_TYPE_INT16:
      return ONNX_TYPE_INT16;
    case TRITONSERVER_TYPE_INT32:
      return ONNX_TYPE_INT32;
    case TRITONSERVER_TYPE_INT64:
      return ONNX_TYPE_INT64;
    case TRITONSERVER_TYPE_FP32:
      return ONNX_TYPE_FLOAT;
    case TRITONSERVER_TYPE_FP64:
      return ONNX_TYPE_DOUBLE;
    case TRITONSERVER_TYPE_BYTES:
      return ONNX_TYPE_STRING;
    default:
      break;
  }
}

}}}  // namespace triton::backend::onnxmlir
