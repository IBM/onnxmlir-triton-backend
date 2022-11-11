#include "model_state.h"
#include "onnxmlir_utils.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace onnxmlir {

ModelState::ModelState(TRITONBACKEND_Model* triton_model): BackendModel(triton_model){ 
    input_tensors = ReadTensorConfig("input");
    output_tensors = ReadTensorConfig("output");
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state){
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

std::vector<TensorDef> ModelState::ReadTensorConfig(char *member){
  std::vector<TensorDef> ret;
  common::TritonJson::Value tensors;
  THROW_IF_BACKEND_MODEL_ERROR(ModelConfig().MemberAsArray(member, &tensors));
  for(size_t i = 0; i< tensors.ArraySize(); i++){
    common::TritonJson::Value tensor;
    THROW_IF_BACKEND_MODEL_ERROR(tensors.IndexAsObject(i, &tensor));
    TensorDef tensor_def(tensor);
    ret.push_back(tensor_def);
  }
  return ret;
}

TensorDef::TensorDef(triton::common::TritonJson::Value &tensor){
    THROW_IF_BACKEND_MODEL_ERROR(tensor.MemberAsString("name", &name));
    std::string member;
    THROW_IF_BACKEND_MODEL_ERROR(tensor.MemberAsString("data_type", &member));
    triton_dtype = ModelConfigDataTypeToTritonServerDataType(member);
    dtype_size = TRITONSERVER_DataTypeByteSize(triton_dtype);
    om_dtype = TritonDataTypeToOmDataType(triton_dtype);
    if(!om_dtype)
      throw triton::backend::BackendModelException(TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, ("No ONNX MLIR datatype for " + member).c_str()));
    triton::common::TritonJson::Value reshape;
    if (tensor.Find("reshape", &reshape)) {
      THROW_IF_BACKEND_MODEL_ERROR(ParseShape(reshape, "shape", &shape));
    } else {
      THROW_IF_BACKEND_MODEL_ERROR(ParseShape(tensor, "dims", &shape));
    }
    size = shape[0];
    for(size_t i = 1; i < shape.size(); i++){
      size *= shape[i];
    }
    byte_size = size * dtype_size;
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