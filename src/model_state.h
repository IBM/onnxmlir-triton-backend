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

#ifndef ONNX_MLIR_MODEL_STATE_H
#define ONNX_MLIR_MODEL_STATE_H
 
#include <vector>
#include "triton/backend/backend_model.h"

#include <OnnxMlirRuntime.h>


namespace triton { namespace backend { namespace onnxmlir {

class ModelState;

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
    bool CheckTensorMatches(ModelState *model_state, OMTensor *tensor, std::string &error);
    bool CheckSignature(const rapidjson::Value &signature, std::string &error);
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
  virtual ~ModelState();
  std::vector<TensorDef> input_tensors;
  std::vector<TensorDef> output_tensors;
  bool supports_first_dim_batching;
  const char* const* (*dll_omQueryEntryPoints)(int64_t*);
  const char* (*dll_omInputSignature)(const char *);
  const char* (*dll_omOutputSignature)(const char *);
  OMTensorList* (*dll_run_main_graph)(OMTensorList *);
  OMTensor* (*dll_omTensorCreate)(void *, int64_t *, int64_t, OM_DATA_TYPE);
  OMTensorList *(*dll_omTensorListCreate)(OMTensor **, int);
  OMTensor* (*dll_omTensorListGetOmtByIndex)(OMTensorList *, int64_t);
  void* (*dll_omTensorGetDataPtr)(OMTensor *);
  int64_t (*dll_omTensorGetRank)(OMTensor *);
  int64_t* (*dll_omTensorGetShape)(OMTensor *);
  OM_DATA_TYPE (*dll_omTensorGetDataType)(OMTensor *);
  void (*dll_omTensorDestroy)(OMTensor *tensor);
  int64_t (*dll_omTensorListGetSize)(OMTensorList *);
  void (*dll_omTensorListDestroy)(OMTensorList *);

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  std::vector<TensorDef> ReadTensorConfig(const char *member);
  TRITONSERVER_Error* LoadModel();
  void *model_lib = nullptr;
};

}}}  // namespace triton::backend::onnxmlir

#endif //ONNX_MLIR_MODEL_STATE_H
