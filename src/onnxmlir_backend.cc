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
      {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

  const size_t num_inputs = model_state->input_names_.size();
  OMTensor *om_inputs[num_inputs];

  for(size_t i = 0; i < num_inputs; i++){
    const char* input_buffer;
    size_t input_buffer_byte_size;
    TRITONSERVER_MemoryType input_buffer_memory_type;
    int64_t input_buffer_memory_type_id;

    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(
            model_state->input_names_[i].c_str(), nullptr /* existing_buffer */,
            0 /* existing_buffer_byte_size */, allowed_input_types, &input_buffer,
            &input_buffer_byte_size, &input_buffer_memory_type,
            &input_buffer_memory_type_id));
    
    TRITONBACKEND_Input* input;
    TRITONBACKEND_RequestInput(requests[0], model_state->input_names_[i].c_str(), &input);
    const int64_t* shape_ptr;
    uint32_t dims_count;
    TRITONSERVER_DataType datatype;
    TRITONBACKEND_InputProperties(input, nullptr, &datatype, &shape_ptr, &dims_count, nullptr, nullptr);
    int64_t shape[dims_count];
    memccpy(shape,shape_ptr, dims_count, sizeof(int64_t));
    int64_t request_size = 1;
    for(size_t i = 1; i < dims_count; i++)
      request_size *= shape[i];
    uint32_t dt_size = TRITONSERVER_DataTypeByteSize(datatype);
    shape[0] = input_buffer_byte_size / dt_size / request_size;
    OM_DATA_TYPE om_dtype = TritonDataTypeToOmDataType(datatype);
    RESPOND_ALL_AND_SET_NULL_IF_FALSE(om_dtype);
    omTensorCreate
  }

  // Finalize the collector. If 'true' is returned, 'input_buffer'
  // will not be valid until the backend synchronizes the CUDA
  // stream or event that was used when creating the collector. For
  // this backend, GPU is not supported and so no CUDA sync should
  // be needed; so if 'true' is returned simply log an error.
  const bool need_cuda_input_sync = collector.Finalize();
  if (need_cuda_input_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'minimal' backend: unexpected CUDA sync required by collector");
  }

  // 'input_buffer' contains the batched "IN0" tensor. The backend can
  // implement whatever logic is necesary to produce "OUT0". This
  // backend simply returns the IN0 value in OUT0 so no actual
  // computation is needed.

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ": requests in batch " +
       std::to_string(request_count))
          .c_str());
  std::string tstr;
  IGNORE_ERROR(BufferAsTypedString(
      tstr, input_buffer, input_buffer_byte_size, TRITONSERVER_TYPE_INT32));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("batched IN0 value: ") + tstr).c_str());

  //const char* output_buffer = input_buffer;
  //TRITONSERVER_MemoryType output_buffer_memory_type = input_buffer_memory_type;
  //int64_t output_buffer_memory_type_id = input_buffer_memory_type_id;

  // This backend supports models that batch along the first dimension
  // and those that don't batch. For non-batch models the output shape
  // will be [ 4 ]. For batch models the output shape will be [ -1, 4
  // ] and the backend "responder" utility below will set the
  // appropriate batch dimension value for each response.
  std::vector<int64_t> output_batch_shape;
  bool supports_first_dim_batching;
  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      model_state->SupportsFirstDimBatching(&supports_first_dim_batching));
  if (supports_first_dim_batching) {
    output_batch_shape.push_back(-1);
  }
  output_batch_shape.push_back(4);

  // Because the OUT0 values are concatenated into a single contiguous
  // 'output_buffer', the backend must "scatter" them out to the
  // individual response OUT0 tensors.  The backend utilities provide
  // a "responder" to facilitate this scattering process.

  // The 'responders's ProcessTensor function will copy the portion of
  // 'output_buffer' corresonding to each request's output into the
  // response for that request.

  BackendOutputResponder responder(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      supports_first_dim_batching, false /* pinned_enabled */,
      nullptr /* stream*/);

  responder.ProcessTensor(
      "OUT0", TRITONSERVER_TYPE_INT32, output_batch_shape, output_buffer,
      output_buffer_memory_type, output_buffer_memory_type_id);

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
        "'minimal' backend: unexpected CUDA sync required by responder");
  }

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

}}}  // namespace triton::backend::minimal
