// Copyright contributors to the onnxmlir-triton-backend project

// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "model_state.h"
#include "onnxmlir_utils.h"
#include "triton/core/tritonbackend.h"

#include "rapidjson/document.h"

#include <dlfcn.h>

namespace triton { namespace backend { namespace onnxmlir {

TensorDef::TensorDef(triton::common::TritonJson::Value &tensor, bool supports_first_dim_batching){
    THROW_IF_BACKEND_MODEL_ERROR(tensor.MemberAsString("name", &name));
    std::string member;
    THROW_IF_BACKEND_MODEL_ERROR(tensor.MemberAsString("data_type", &member));
    triton_dtype = ModelConfigDataTypeToTritonServerDataType(member);
    dtype_size = TRITONSERVER_DataTypeByteSize(triton_dtype);
    om_dtype = TritonDataTypeToOmDataType(triton_dtype);
    if(om_dtype == ONNX_TYPE_UNDEFINED)
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
    if(supports_first_dim_batching)
      shape.insert(shape.begin(), -1);
}

bool TensorDef::CheckTensorMatches(ModelState *model_state, OMTensor *tensor, std::string &error){
  OM_DATA_TYPE tensor_dt = model_state->dll_omTensorGetDataType(tensor);
  if(tensor_dt != om_dtype){
    error = "datatype missmatches config";
  }
  int64_t tensor_dims = shape.size();
  if(tensor_dims != model_state->dll_omTensorGetRank(tensor)){
    error = "number of dimensions missmatches config: " + std::to_string(shape.size()) + " actual: " + std::to_string(tensor_dims);
    return false;
  }
  int64_t *tensor_shape = model_state->dll_omTensorGetShape(tensor);
  for(int64_t s = model_state->supports_first_dim_batching ? 1:0; s < tensor_dims; s++){
    if(tensor_shape[s] != shape[s]){
      std::string shape_str;
      IGNORE_ERROR(BufferAsTypedString(shape_str, (const char*)tensor_shape, tensor_dims * sizeof(int64_t), TRITONSERVER_TYPE_INT64));
      error = "shape missmatches config: " + shape_str;
      return false;
    }
  }
  return true;
}

bool TensorDef::CheckSignature(const rapidjson::Value &signature, std::string &error){
  if(signature["name"].GetString() != name){
    error = "name";
    return false;
  }
  OM_DATA_TYPE type = MlirDataTypeToOmDataType(signature["type"].GetString());
  if(om_dtype != type){
    error = "type";
    return false;
  }
  const rapidjson::Value& dims = signature["dims"];
  if(dims.Size() != shape.size()){
    error = "rank";
    return false;
  }
  for(rapidjson::SizeType j = 0; j < dims.Size(); j++){
    if(dims[j].GetInt64() != shape[j]){
      error = "shape";
      return false;
    }
  }
  return true;
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model): BackendModel(triton_model){
  THROW_IF_BACKEND_MODEL_ERROR(SupportsFirstDimBatching(&supports_first_dim_batching));
  input_tensors = ReadTensorConfig("input");
  output_tensors = ReadTensorConfig("output");
  THROW_IF_BACKEND_MODEL_ERROR(LoadModel());
}

ModelState::~ModelState(){
  if(model_lib)
    dlclose(model_lib);
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

std::vector<TensorDef> ModelState::ReadTensorConfig(const char *member){
  std::vector<TensorDef> ret;
  common::TritonJson::Value tensors;
  THROW_IF_BACKEND_MODEL_ERROR(ModelConfig().MemberAsArray(member, &tensors));
  for(size_t i = 0; i< tensors.ArraySize(); i++){
    common::TritonJson::Value tensor;
    THROW_IF_BACKEND_MODEL_ERROR(tensors.IndexAsObject(i, &tensor));
    TensorDef tensor_def(tensor, supports_first_dim_batching);
    ret.push_back(tensor_def);
  }
  return ret;
}

bool CheckSignature(const char *signature, std::vector<TensorDef> &config, std::string &error){
  rapidjson::Document d;
  d.Parse(signature);
  if(d.HasParseError()){
    error = "Signature Parse Error";
    return false;
  }
  if(d.Size() != config.size()){
    error = "number of tensors";
    return false;
  }
  for (rapidjson::SizeType i = 0; i < d.Size(); i++){
      const rapidjson::Value& tensor = d[i];
      if(!config[i].CheckSignature(tensor, error))
        return false;
  }
  return true;
}

#define RETURN_DLERROR_IF_NULL(x) RETURN_ERROR_IF_FALSE(x, TRITONSERVER_ERROR_UNAVAILABLE, std::string(dlerror()))

TRITONSERVER_Error*
ModelState::LoadModel(){
  std::string so_model_filename = "model.so";
  std::string model_path = JoinPath({ RepositoryPath(), std::to_string(Version()), so_model_filename});
  {
    bool exists;
    RETURN_IF_ERROR(FileExists(model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + model_path + "' for model '" +
            Name() + "'");
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,("Loading " + model_path).c_str());
  model_lib = dlopen(model_path.c_str(), RTLD_LAZY);
  RETURN_ERROR_IF_FALSE(model_lib, TRITONSERVER_ERROR_UNAVAILABLE, std::string("failed to load ") + model_path + ": " + dlerror());
  dll_omQueryEntryPoints = (const char* const* (*)(int64_t*)) dlsym(model_lib, "omQueryEntryPoints");
  RETURN_DLERROR_IF_NULL(dll_omQueryEntryPoints);
  dll_omInputSignature = (const char* (*)(const char *)) dlsym(model_lib, "omInputSignature");
  RETURN_DLERROR_IF_NULL(dll_omInputSignature);
  dll_omOutputSignature = (const char* (*)(const char *)) dlsym(model_lib, "omOutputSignature");
  RETURN_DLERROR_IF_NULL(dll_omOutputSignature);
  
  int64_t num_entry_points;
  const char* const* entry_points = dll_omQueryEntryPoints(&num_entry_points);
  const char *entry_point = "run_main_graph";
  bool found = false;
  for(int64_t i=0; i < num_entry_points; i++){
    if(strcmp(entry_point, entry_points[i]))
      continue;
    std::string input_sig(dll_omInputSignature(entry_points[i]));
    std::string output_sig(dll_omOutputSignature(entry_points[i]));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,("entrypoint: " + std::string(entry_points[i]) 
                                        + "\n input:\n" + input_sig
                                        + "\n output:\n" + output_sig ).c_str());
    std::string error;                                        
    RETURN_ERROR_IF_FALSE(
        CheckSignature(input_sig.c_str(), input_tensors, error),
        TRITONSERVER_ERROR_UNAVAILABLE,
        "input signature for entry point '" + std::string(entry_point) + " for model '" +
            Name() + "' mismatches config: " + error);
    RETURN_ERROR_IF_FALSE(
        CheckSignature(output_sig.c_str(), output_tensors, error),
        TRITONSERVER_ERROR_UNAVAILABLE,
        "output signature for entry point '" + std::string(entry_point) + " for model '" +
            Name() + "' mismatches config: " + error);
    found = true;
    break;
  }
  RETURN_ERROR_IF_FALSE(
        found, TRITONSERVER_ERROR_UNAVAILABLE,
        "unable to find entry point '" + std::string(entry_point) + " for model '" +
            Name() + "'");
  dll_run_main_graph = (OMTensorList * (*)(OMTensorList *)) dlsym(model_lib, entry_point);
  RETURN_DLERROR_IF_NULL(dll_run_main_graph);
  dll_omTensorCreate = (OMTensor * (*)(void *, int64_t *, int64_t, OM_DATA_TYPE)) dlsym(model_lib, "omTensorCreate");
  RETURN_DLERROR_IF_NULL(dll_omTensorCreate);
  dll_omTensorListCreate = (OMTensorList * (*)(OMTensor **, int)) dlsym(model_lib, "omTensorListCreate");
  RETURN_DLERROR_IF_NULL(dll_omTensorListCreate);
  dll_omTensorListGetOmtByIndex = (OMTensor * (*)(OMTensorList *, int64_t)) dlsym(model_lib, "omTensorListGetOmtByIndex");
  RETURN_DLERROR_IF_NULL(dll_omTensorListGetOmtByIndex);
  dll_omTensorGetDataPtr = (void* (*)(OMTensor *))dlsym(model_lib, "omTensorGetDataPtr");
  RETURN_DLERROR_IF_NULL( dll_omTensorGetDataPtr);
  dll_omTensorGetRank = (int64_t (*)(OMTensor *))dlsym(model_lib, "omTensorGetRank");
  RETURN_DLERROR_IF_NULL( dll_omTensorGetRank);
  dll_omTensorGetShape = (int64_t* (*)(OMTensor *))dlsym(model_lib, "omTensorGetShape");
  RETURN_DLERROR_IF_NULL( dll_omTensorGetShape);
  dll_omTensorGetDataType = (OM_DATA_TYPE (*)(OMTensor *))dlsym(model_lib, "omTensorGetDataType");
  RETURN_DLERROR_IF_NULL( dll_omTensorGetDataType);
  dll_omTensorListGetSize = (int64_t (*)(OMTensorList *))dlsym(model_lib, "omTensorListGetSize");
  RETURN_DLERROR_IF_NULL(dll_omTensorListGetSize);
  dll_omTensorListDestroy = (void (*)(OMTensorList *))dlsym(model_lib, "omTensorListDestroy");
  RETURN_DLERROR_IF_NULL(dll_omTensorListDestroy);
  dll_omTensorDestroy = (void (*)(OMTensor *))dlsym(model_lib, "omTensorDestroy");
  RETURN_DLERROR_IF_NULL(dll_omTensorDestroy);
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
