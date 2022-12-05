// Copyright contributors to the onnxmlir-triton-backend project

#include "onnxmlir_typemapping.h"

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
      return ONNX_TYPE_UNDEFINED;
  }
}

const std::map<std::string, OM_DATA_TYPE> OM_DATA_TYPE_MLIR_TO_ONNX = {
    {"i1", ONNX_TYPE_BOOL},   // bool  -> BOOL
    {"i8", ONNX_TYPE_INT8},   // char  -> INT8 (platform dependent, can be UINT8)
    {"si8", ONNX_TYPE_INT8},   // int8_t   -> INT8
    {"ui8", ONNX_TYPE_UINT8},  // uint8_t  -> UINT8,  unsigned char  -> UNIT 8
    {"i16", ONNX_TYPE_INT16},
    {"si16", ONNX_TYPE_INT16},  // int16_t  -> INT16,  short          -> INT16
    {"ui16", ONNX_TYPE_UINT16}, // uint16_t -> UINT16, unsigned short -> UINT16
    {"i32", ONNX_TYPE_INT32},  // int32_t  -> INT32,  int            -> INT32
    {"si32", ONNX_TYPE_INT32},
    {"ui32", ONNX_TYPE_UINT32}, // uint32_t -> UINT32, unsigned int   -> UINT32
    {"i64", ONNX_TYPE_INT64},  // int64_t  -> INT64,  long           -> INT64
    {"si64", ONNX_TYPE_INT64},
    {"ui64", ONNX_TYPE_UINT64}, // uint64_t -> UINT64, unsigned long  -> UINT64
    {"f32", ONNX_TYPE_FLOAT},  // float    -> FLOAT
    {"f64", ONNX_TYPE_DOUBLE}, // double   -> DOUBLE
    {"!krnl.string", ONNX_TYPE_STRING},    // const char * -> STRING
    {"complex<f32>", ONNX_TYPE_COMPLEX64},  // _Complex float -> COMPLEX64
    {"complex<f64>", ONNX_TYPE_COMPLEX128}, // _Complex double -> COMPLEX128
};

OM_DATA_TYPE MlirDataTypeToOmDataType(std::string datatype){
  auto pos = OM_DATA_TYPE_MLIR_TO_ONNX.find(datatype);
  if(pos == OM_DATA_TYPE_MLIR_TO_ONNX.end())
    return ONNX_TYPE_UNDEFINED;
  return pos->second;
}

}}}  // namespace triton::backend::onnxmlir
