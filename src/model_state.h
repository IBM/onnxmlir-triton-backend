#ifndef ONNX_MLIR_MODEL_STATE_H
#define ONNX_MLIR_MODEL_STATE_H
 
#include <vector>
#include "triton/backend/backend_model.h"

#include <OnnxMlirRuntime.h>


namespace triton { namespace backend { namespace onnxmlir {

class TensorDef {
  public:
    std::string name;
    std::vector<int64_t> shape;
    int64_t size;
    OM_DATA_TYPE om_dtype;
    TRITONSERVER_DataType triton_dtype;
    uint32_t dtype_size;
    int64_t byte_size;
    TensorDef(triton::common::TritonJson::Value &tensor, bool supports_first_dim_batching);   
};

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;
  std::vector<TensorDef> input_tensors;
  std::vector<TensorDef> output_tensors;
  bool supports_first_dim_batching;
  OMTensorList* (*dll_run_main_graph)(OMTensorList *);
  OMTensor* (*dll_omTensorCreate)(void *, int64_t *, int64_t, OM_DATA_TYPE);
  OMTensorList *(*dll_omTensorListCreate)(OMTensor **, int);
  OMTensor* (*dll_omTensorListGetOmtByIndex)(OMTensorList *, int64_t);
  void* (*dll_omTensorGetDataPtr)(OMTensor *);
  int64_t (*dll_omTensorGetRank)(OMTensor *);
  int64_t* (*dll_omTensorGetShape)(OMTensor *);
  void (*dll_omTensorDestroy)(OMTensor *tensor);
  int64_t (*dll_omTensorListGetSize)(OMTensorList *);
  void (*dll_omTensorListDestroy)(OMTensorList *);

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  std::vector<TensorDef> ReadTensorConfig(const char *member);
  TRITONSERVER_Error* LoadModel();
  void *model_lib;
};

}}}  // namespace triton::backend::onnxmlir

#endif //ONNX_MLIR_MODEL_STATE_H