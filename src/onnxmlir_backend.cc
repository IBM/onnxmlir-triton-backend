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

#include "model_instance_state.h"
#include "onnxmlir_utils.h"

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include <OnnxMlirRuntime.h>

namespace triton { namespace backend { namespace onnxmlir {

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Best practice for a high-performance
  // implementation is to avoid introducing mutex/lock and instead use
  // only function-local and model-instance-specific state.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // 'responses' is initialized as a parallel array to 'requests',
  // with one TRITONBACKEND_Response object for each
  // TRITONBACKEND_Request object. If something goes wrong while
  // creating these response objects, the backend simply returns an
  // error from TRITONBACKEND_ModelInstanceExecute, indicating to
  // Triton that this backend did not create or send any responses and
  // so it is up to Triton to create and send an appropriate error
  // response for each request. RETURN_IF_ERROR is one of several
  // useful macros for error handling that can be found in
  // backend_common.h.

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // At this point, the backend takes ownership of 'requests', which
  // means that it is responsible for sending a response for every
  // request. From here, even if something goes wrong in processing,
  // the backend must return 'nullptr' from this function to indicate
  // success. Any errors and failures must be communicated via the
  // response objects.
  //
  // To simplify error handling, the backend utilities manage
  // 'responses' in a specific way and it is recommended that backends
  // follow this same pattern. When an error is detected in the
  // processing of a request, an appropriate error response is sent
  // and the corresponding TRITONBACKEND_Response object within
  // 'responses' is set to nullptr to indicate that the
  // request/response has already been handled and no futher processing
  // should be performed for that request. Even if all responses fail,
  // the backend still allows execution to flow to the end of the
  // function. RESPOND_AND_SET_NULL_IF_ERROR, and
  // RESPOND_ALL_AND_SET_NULL_IF_ERROR are macros from
  // backend_common.h that assist in this management of response
  // objects.

  // The backend could iterate over the 'requests' and process each
  // one separately. But for performance reasons it is usually
  // preferred to create batched input tensors that are processed
  // simultaneously. This is especially true for devices like GPUs
  // that are capable of exploiting the large amount parallelism
  // exposed by larger data sets.
  //
  // The backend utilities provide a "collector" to facilitate this
  // batching process. The 'collector's ProcessTensor function will
  // combine a tensor's value from each request in the batch into a
  // single contiguous buffer. The buffer can be provided by the
  // backend or 'collector' can create and manage it. In this backend,
  // there is not a specific buffer into which the batch should be
  // created, so use ProcessTensor arguments that cause collector to
  // manage it.

  BackendInputCollector collector(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      false /* pinned_enabled */, nullptr /* stream*/);

  // To instruct ProcessTensor to "gather" the entire batch of IN0
  // input tensors into a single contiguous buffer in CPU memory, set
  // the "allowed input types" to be the CPU ones (see tritonserver.h
  // in the triton-inference-server/core repo for allowed memory
  // types).
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types =
      {{TRITONSERVER_MEMORY_CPU_PINNED, 0}};

  const size_t num_inputs = model_state->input_tensors.size();

  // will be freed by omTensorListDestroy()
  OMTensor **om_inputs = (OMTensor **) malloc(num_inputs * sizeof(OMTensor *));

  for(size_t i = 0; i < num_inputs; i++){
    TensorDef input_def = model_state->input_tensors[i];
    const char* input_buffer;
    size_t input_buffer_byte_size;
    TRITONSERVER_MemoryType input_buffer_memory_type;
    int64_t input_buffer_memory_type_id;

    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(
            input_def.name.c_str(), nullptr /* existing_buffer */,
            0 /* existing_buffer_byte_size */, allowed_input_types, &input_buffer,
            &input_buffer_byte_size, &input_buffer_memory_type,
            &input_buffer_memory_type_id));
    
    //TRITONBACKEND_Input* input;
    //TRITONBACKEND_RequestInput(requests[0], input_def.name.c_str(), &input);
    //const int64_t* shape_ptr;
    //uint32_t dims_count;
    //TRITONSERVER_DataType datatype;
    //TRITONBACKEND_InputProperties(input, nullptr, &datatype, &shape_ptr, &dims_count, nullptr, nullptr);
    int64_t in_shape[input_def.shape.size()];
    std::copy(input_def.shape.begin(), input_def.shape.end(), in_shape);
    if(model_state->supports_first_dim_batching)
      in_shape[0] = input_buffer_byte_size / input_def.byte_size;
    om_inputs[i] = instance_state->dll_omTensorCreate((void* )input_buffer, in_shape, input_def.shape.size(), input_def.om_dtype);
  }

  OMTensorList *om_input_tl = instance_state->dll_omTensorListCreate(om_inputs, num_inputs);

  // Finalize the collector. If 'true' is returned, 'input_buffer'
  // will not be valid until the backend synchronizes the CUDA
  // stream or event that was used when creating the collector. For
  // this backend, GPU is not supported and so no CUDA sync should
  // be needed; so if 'true' is returned simply log an error.
  const bool need_cuda_input_sync = collector.Finalize();
  if (need_cuda_input_sync) {
    instance_state->dll_omTensorListDestroy(om_input_tl);
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'onnxmlir' backend: unexpected CUDA sync required by collector");
  }

  //Run the Model
  OMTensorList *om_output_tl = instance_state->dll_run_main_graph(om_input_tl);

  instance_state->dll_omTensorListDestroy(om_input_tl);
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ": requests in batch " +
       std::to_string(request_count))
          .c_str());

  int64_t config_output_size = model_state->output_tensors.size();
  int64_t output_size = instance_state->dll_omTensorListGetSize(om_output_tl);
  if(output_size != config_output_size){
    instance_state->dll_omTensorListDestroy(om_output_tl);
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, 
      ("Number of ouput Tensors missmatches config: " + std::to_string(config_output_size) + " actual: " + std::to_string(output_size)).c_str()));
  }

  // Because the output values are concatenated into a single contiguous
  // 'output_buffer', the backend must "scatter" them out to the
  // individual response output tensors.  The backend utilities provide
  // a "responder" to facilitate this scattering process.

  // The 'responders's ProcessTensor function will copy the portion of
  // 'output_buffer' corresonding to each request's output into the
  // response for that request.

  BackendOutputResponder responder(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      model_state->supports_first_dim_batching, false /* pinned_enabled */,
      nullptr /* stream*/);

  for(int64_t i = 0; i < output_size; i++){
    TensorDef output_def = model_state->output_tensors[i];
    OMTensor *om_output = instance_state->dll_omTensorListGetOmtByIndex(om_output_tl, i);
    void *output_buffer = instance_state->dll_omTensorGetDataPtr(om_output);
    int64_t out_dims = output_def.shape.size();
    if(out_dims != instance_state->dll_omTensorGetRank(om_output)){
      instance_state->dll_omTensorListDestroy(om_output_tl);
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, 
      ("Number of ouput dimensions missmatches config: " + std::to_string(output_def.shape.size()) + " actual: " + std::to_string(out_dims)).c_str()));
    }
    int64_t *out_shape = instance_state->dll_omTensorGetShape(om_output);
    for(int64_t s = 0; s < out_dims; s++){
      if(out_shape[s + 1] != output_def.shape[s]){
        instance_state->dll_omTensorListDestroy(om_output_tl);
        RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, 
        ("ouput shapes missmatch config")));
      }
    }
    //Process tensor might modify output_shape, so we copy it
    auto output_shape = output_def.shape;
    responder.ProcessTensor(
    output_def.name, TRITONSERVER_TYPE_INT32, output_shape, (const char*)output_buffer,
    TRITONSERVER_MEMORY_CPU_PINNED, 0);
  }

  // Finalize the responder. If 'true' is returned, the OUT0
  // tensors' data will not be valid until the backend synchronizes
  // the CUDA stream or event that was used when creating the
  // responder. For this backend, GPU is not supported and so no
  // CUDA sync should be needed; so if 'true' is returned simply log
  // an error.
  const bool need_cuda_output_sync = responder.Finalize();
  if (need_cuda_output_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'onnxmlir' backend: unexpected CUDA sync required by responder");
  }

  instance_state->dll_omTensorListDestroy(om_output_tl);

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send response");
    }
  }

  // Done with the request objects so release them.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::onnxmlir
