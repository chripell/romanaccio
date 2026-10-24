cmake_minimum_required(VERSION 3.16)
project(eswin_runtime VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

find_package(OpenMP)

# Core Runtime Library
set(RUNTIME_SOURCES
    src/backend/driver_wrapper.cpp
    src/backend/dsp_driver.cpp
    src/backend/dma_buffer.cpp
    src/backend/worker_pool.cpp
    src/backend/pipeline_executor.cpp
    src/model/sha256.cpp
    src/model/model_manifest.cpp
    src/model/embedding_table.cpp
    src/model/safetensors_loader.cpp
    src/model/downloader.cpp
    src/nlp/sampler.cpp
    src/nlp/tokenizer.cpp
    src/llm/kv_cache.cpp
    src/llm/hardware_kv_cache.cpp
    src/llm/transformer_session.cpp
    src/api/es_runtime.cpp
)

add_library(es_runtime SHARED ${RUNTIME_SOURCES})
target_link_libraries(es_runtime PRIVATE ${CMAKE_DL_LIBS} pthread)
if(OpenMP_CXX_FOUND)
    target_link_libraries(es_runtime PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(es_runtime PRIVATE ${OpenMP_CXX_FLAGS})
endif()
target_include_directories(es_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Milestone 1 DMA Test Executable
add_executable(test_dma_alloc tests/test_dma_alloc.cpp)
target_link_libraries(test_dma_alloc PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 2 Concurrency Test Executable
add_executable(test_dual_die_concurrency tests/test_dual_die_concurrency.cpp)
target_link_libraries(test_dual_die_concurrency PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 3 Manifest Test Executable
add_executable(test_manifest tests/test_manifest.cpp)
target_link_libraries(test_manifest PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 3 Fetch Tool Executable
add_executable(eswin-fetch tools/eswin_fetch.cpp)
target_link_libraries(eswin-fetch PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 4 Tokenizer & Sampler Test Executable
add_executable(test_tokenizer_sampler tests/test_tokenizer_sampler.cpp)
target_link_libraries(test_tokenizer_sampler PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 5 LLM Session Test Executable
add_executable(test_transformer_session tests/test_transformer_session.cpp)
target_link_libraries(test_transformer_session PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 6 End-to-End Test Executable
add_executable(test_runtime_e2e tests/test_runtime_e2e.cpp)
target_link_libraries(test_runtime_e2e PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 7 Multi-Layer Block Chaining Test Executable
add_executable(test_layer_chaining tests/test_layer_chaining.cpp)
target_link_libraries(test_layer_chaining PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 8 Embedding & Weights Loader Test Executable
add_executable(test_embedding_loader tests/test_embedding_loader.cpp)
target_link_libraries(test_embedding_loader PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 9 Hardware MMZ KV-Cache Test Executable
add_executable(test_hardware_kv_cache tests/test_hardware_kv_cache.cpp)
target_link_libraries(test_hardware_kv_cache PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 10 DeepSeek/Qwen2 End-to-End Test Executable
add_executable(test_deepseek_e2e tests/test_deepseek_e2e.cpp)
target_link_libraries(test_deepseek_e2e PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# Milestone 6 Interactive Chat Tool Executable
add_executable(eswin-chat tools/eswin_chat.cpp)
target_link_libraries(eswin-chat PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

# DSP Core Runner Tool Executable
add_executable(es_run_dsp tools/es_run_dsp.cpp)
target_include_directories(es_run_dsp PRIVATE src)
target_link_libraries(es_run_dsp PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

enable_testing()
add_test(NAME test_dma_alloc COMMAND test_dma_alloc)
add_test(NAME test_dual_die_concurrency COMMAND test_dual_die_concurrency 20)
add_test(NAME test_manifest COMMAND test_manifest)
add_test(NAME test_tokenizer_sampler COMMAND test_tokenizer_sampler)
add_test(NAME test_transformer_session COMMAND test_transformer_session)
add_test(NAME test_runtime_e2e COMMAND test_runtime_e2e)
add_test(NAME test_layer_chaining COMMAND test_layer_chaining)
add_test(NAME test_embedding_loader COMMAND test_embedding_loader)
add_test(NAME test_hardware_kv_cache COMMAND test_hardware_kv_cache)
add_test(NAME test_deepseek_e2e COMMAND test_deepseek_e2e)

