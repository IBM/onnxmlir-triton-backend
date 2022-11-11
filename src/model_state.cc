#include "model_state.h"
#include "onnxmlir_utils.h"

namespace triton { namespace backend { namespace onnxmlir {

ModelState::ModelState(TRITONBACKEND_Model* triton_model): BackendModel(triton_model) { 
    THROW_IF_BACKEND_MODEL_ERROR(ReadTensorConfig("input", &input_names_, &input_dtypes_));
    THROW_IF_BACKEND_MODEL_ERROR(ReadTensorConfig("output", &output_names_, &output_dtypes_));
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error* ModelState::ReadTensorConfig(char *member, std::vector<std::string> *names, std::vector<OM_DATA_TYPE> *dtypes){
    common::TritonJson::Value tensors;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray(member, &tensors));
    for(size_t i = 0; i< tensors.ArraySize(); i++){
      common::TritonJson::Value tensor;
      RETURN_IF_ERROR(tensors.IndexAsObject(i, &tensor));
      std::string member;
      RETURN_IF_ERROR(tensor.MemberAsString("name", &member));
      names->push_back(member);
      RETURN_IF_ERROR(tensor.MemberAsString("data_type", &member));
      OM_DATA_TYPE dtype = TritonDataTypeToOmDataType(ModelConfigDataTypeToTritonServerDataType(member));
      RETURN_ERROR_IF_FALSE(dtype, TRITONSERVER_ERROR_UNSUPPORTED, "No ONNX MLIR datatype for " + member);
      dtypes->push_back(dtype);

    }
  return nullptr;
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::onnxmlir