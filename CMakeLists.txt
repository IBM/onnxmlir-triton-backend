# Copyright contributors to the onnxmlir-triton-backend project

# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(onnxmlirbackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because onnxmlir backend doesn't
# use GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(ONNX_MLIR_REPO_TAG "main" CACHE STRING "Tag for onnx/onnx-mlir repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-onnxmlir
  GIT_REPOSITORY https://github.com/onnx/onnx-mlir.git
  GIT_TAG ${ONNX_MLIR_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)
FetchContent_Populate(repo-onnxmlir)
include_directories(${repo-onnxmlir_SOURCE_DIR}/include)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_onnxmlir.ldscript libtriton_onnxmlir.ldscript COPYONLY)

add_library(
  triton-onnxmlir-backend SHARED
  src/onnxmlir_backend.cc
  src/model_state.cc
  src/model_instance_state.cc
  src/onnxmlir_utils.cc
)

add_library(
  OnnxMlirBackend::triton-onnxmlir-backend ALIAS triton-onnxmlir-backend
)

target_include_directories(
  triton-onnxmlir-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  repo-onnxmlir
)

target_compile_features(triton-onnxmlir-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-onnxmlir-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

target_link_libraries(
  triton-onnxmlir-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

if(WIN32)
  set_target_properties(
    triton-onnxmlir-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_onnxmlir
  )
else()
  set_target_properties(
    triton-onnxmlir-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_onnxmlir
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_onnxmlir.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_onnxmlir.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/OnnxMlirBackend)

install(
  TARGETS
    triton-onnxmlir-backend
  EXPORT
    triton-onnxmlir-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxmlir
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxmlir
)

install(
  EXPORT
    triton-onnxmlir-backend-targets
  FILE
    OnnxMlirBackendTargets.cmake
  NAMESPACE
    OnnxMlirBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/OnnxMlirBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/OnnxMlirBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/OnnxMlirBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-onnxmlir-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/OnnxMlirBackendTargets.cmake
  NAMESPACE OnnxMlirBackend::
)

export(PACKAGE OnnxMlirBackend)
