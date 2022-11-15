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

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state) { }
  ModelState* model_state_;
};

}}}  // namespace triton::backend::onnxmlir

#endif //ONNX_MLIR_MODEL_INSTANCE_STATE_H