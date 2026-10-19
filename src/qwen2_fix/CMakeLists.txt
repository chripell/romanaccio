# Copyright (c) 2023 ESWIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
cmake_minimum_required(VERSION 3.16.1)

project(es_qwen2_demo)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (HOST_C_COMPLIER)
    set(CMAKE_C_COMPILER ${HOST_C_COMPLIER})
endif()
if (HOST_CXX_COMPLIER)
    set(CMAKE_CXX_COMPILER ${HOST_CXX_COMPLIER})
endif()
if (HOST_COMPLIER_FLAGS)
    set(CMAKE_C_FLAGS "${HOST_COMPLIER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${HOST_COMPLIER_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ES_QWEN2_DEMO_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_library(THREAD_LIB pthread)
if (THREAD_LIB)
    message(STATUS "Found pthread: ${THREAD_LIB}")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    set(CMAKE_THREAD_LIBS_INIT "${THREAD_LIB}")
    set(CMAKE_USE_PTHREADS_INIT TRUE)
else()
    message(FATAL_ERROR "pthread not found!")
endif ()

# third-party libraries
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(THIRD_PARTY_DIR ${ES_QWEN2_DEMO_DIR}/third_party)
add_subdirectory(${THIRD_PARTY_DIR}/abseil-cpp)
add_subdirectory(${THIRD_PARTY_DIR}/re2)

message(STATUS "Using ES_QWEN2_DEMO_DIR: ${ES_QWEN2_DEMO_DIR}")
message(STATUS "Using CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "Using CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "Using CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "Using CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

if (NOT DEFINED BUILD_USE_LOCAL_ENV)
    set(JSON_DIR ${NPU_SDK_THIRD_PARTY_BUILD}/json)
else()
    set(JSON_DIR ${THIRD_PARTY_DIR}/json)
endif()

set(TOKENIZER_DIR ${ES_QWEN2_DEMO_DIR}/tokenizer)

include_directories(
    ${ES_QWEN2_DEMO_DIR}
    ${JSON_DIR}/include
    ${TOKENIZER_DIR})

if (NOT DEFINED BUILD_USE_LOCAL_ENV)
    include_directories(
        ${NPU_SDK_ESWIN_HOST}/runtime/include
        ${NPU_SDK_ESWIN_UTIL}
        ${WIN2030_ESSDK_INCLUDE})
else()
    include_directories(
        ${HOST_SYSROOT}/usr/include/essdk
        ${THIRD_PARTY_DIR})
endif()

file(GLOB src_files ${ES_QWEN2_DEMO_DIR}/es_qwen2_client.cpp
     ${ES_QWEN2_DEMO_DIR}/es_qwen2_demo.cpp
     ${TOKENIZER_DIR}/*.cpp
     ${JSON_DIR}/src/*.cpp)

if(NOT DEFINED BUILD_USE_LOCAL_ENV)
    set(ESSDK_MEMORY_LIB $ENV{WIN2030_STAGING_DIR}/usr/lib/essdk)
    link_directories(
        ${NPU_SDK_INSTALL}/riscv64/tests ${WIN2030_ESSDK_LIB}
        ${WIN2030_ESSDK_LIB}/dsp/riscv_64 ${WIN2030_STAGING_DIR}/usr/lib
        ${ESSDK_MEMORY_LIB})
endif()

set(ES_QWEN2_NAME es_qwen2)

add_executable(${ES_QWEN2_NAME} ${src_files})
target_link_libraries(${ES_QWEN2_NAME} PUBLIC -ledla_runtime -les_sys -les_memory)
target_link_libraries(${ES_QWEN2_NAME} PUBLIC -lpthread re2::re2)

file(GLOB src_lib_files ${ES_QWEN2_DEMO_DIR}/es_qwen2_client.cpp
     ${ES_QWEN2_DEMO_DIR}/es_qwen2_process.cpp
     ${TOKENIZER_DIR}/*.cpp
     ${JSON_DIR}/src/*.cpp)

add_library(es_qwen2_process SHARED ${src_lib_files})
target_compile_options(es_qwen2_process PUBLIC -fPIC)
target_link_libraries(es_qwen2_process PUBLIC -ledla_runtime -les_sys -les_memory)
target_link_libraries(es_qwen2_process PUBLIC -lpthread re2::re2)

install(TARGETS ${ES_QWEN2_NAME} RUNTIME DESTINATION .)
