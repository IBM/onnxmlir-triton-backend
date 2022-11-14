#ifndef ONNX_MLIR_MODEL_INSTANCE_STATE_H
#define ONNX_MLIR_MODEL_INSTANCE_STATE_H

#include "triton/backend/backend_model_instance.h"
#include "model_state.h"

#include <OnnxMlirRuntime.h>

namespace triton { namespace backend { namespace onnxmlir {
/////////////
//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

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

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
  {
    LoadModel();
  }
  TRITONSERVER_Error* LoadModel();
  void *model_lib;
  ModelState* model_state_;
};

}}}  // namespace triton::backend::onnxmlir

#endif //ONNX_MLIR_MODEL_INSTANCE_STATE_H