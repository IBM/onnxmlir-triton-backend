#include "model_instance_state.h"
#include "triton/core/tritonbackend.h"
#include <dlfcn.h>

namespace triton { namespace backend { namespace onnxmlir {

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

#define RETURN_DLERROR_IF_NULL(x) RETURN_ERROR_IF_FALSE(x, TRITONSERVER_ERROR_UNAVAILABLE, std::string(dlerror()))

TRITONSERVER_Error*
ModelInstanceState::LoadModel(){
  std::string so_model_filename = ArtifactFilename();
  if(so_model_filename.empty())
    so_model_filename = "model.so";
  std::string model_path = JoinPath({ model_state_->RepositoryPath(), std::to_string(model_state_->Version()), so_model_filename});
  {
    bool exists;
    RETURN_IF_ERROR(FileExists(model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + model_path + "' for model '" +
            Name() + "'");
  }
  model_lib = dlopen(model_path.c_str(), RTLD_LAZY);
  RETURN_ERROR_IF_FALSE(model_lib, TRITONSERVER_ERROR_UNAVAILABLE, std::string("failed to load ") + model_path + ": " + dlerror());
  dll_run_main_graph = (OMTensorList * (*)(OMTensorList *)) dlsym(model_lib, "run_main_graph");
  RETURN_DLERROR_IF_NULL(dll_run_main_graph);
  dll_omTensorCreate = (OMTensor * (*)(void *, int64_t *, int64_t, OM_DATA_TYPE)) dlsym(model_lib, "omTensorCreate");
  RETURN_DLERROR_IF_NULL(dll_omTensorCreate);
  dll_omTensorListCreate = (OMTensorList * (*)(OMTensor **, int)) dlsym(model_lib, "omTensorListCreate");
  RETURN_DLERROR_IF_NULL(dll_omTensorListCreate);
  dll_omTensorListGetOmtByIndex = (OMTensor * (*)(OMTensorList *, int64_t)) dlsym(model_lib, "omTensorListGetOmtByIndex");
  RETURN_DLERROR_IF_NULL(dll_omTensorListGetOmtByIndex);
  dll_omTensorGetDataPtr = (void *(*)(OMTensor *))dlsym(model_lib, "omTensorGetDataPtr");
  RETURN_DLERROR_IF_NULL( dll_omTensorGetDataPtr);
  dll_omTensorListGetSize = (int64_t (*)(OMTensorList *))dlsym(model_lib, "omTensorListGetSize");
  RETURN_DLERROR_IF_NULL(dll_omTensorListDestroy);
  dll_omTensorListDestroy = (void (*)(OMTensorList *))dlsym(model_lib, "omTensorListDestroy");
  RETURN_DLERROR_IF_NULL(dll_omTensorListDestroy);
  dll_omTensorDestroy = (void (*)(OMTensor *))dlsym(model_lib, "omTensorDestroy");
  RETURN_DLERROR_IF_NULL(dll_omTensorDestroy);
  return nullptr;
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  // extern "C"
}}}  // namespace triton::backend::onnxmlir