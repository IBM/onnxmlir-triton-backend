#ifndef ONNX_MLIR_MODEL_STATE_H
#define ONNX_MLIR_MODEL_STATE_H

#include <vector>
#include "triton/backend/backend_model.h"
#include <OnnxMlirRuntime.h>

namespace triton { namespace backend { namespace onnxmlir {

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
  std::vector<std::string> input_names_;
  std::vector<OM_DATA_TYPE> input_dtypes_;
  std::vector<std::string> output_names_;
  std::vector<OM_DATA_TYPE> output_dtypes_;

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* ModelState::ReadTensorConfig(char* member, std::vector<std::string> *names, std::vector<OM_DATA_TYPE> *dtypes);
};

}}}  // namespace triton::backend::onnxmlir

#endif //ONNX_MLIR_MODEL_STATE_H