cmake_minimum_required(VERSION 3.16)
project(eswin_runtime VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

find_package(OpenMP)

# Core Runtime Library
set(RUNTIME_SOURCES
    src/backend/driver_wrapper.cpp
    src/backend/dma_buffer.cpp
    src/backend/worker_pool.cpp
    src/model/sha256.cpp
    src/model/model_manifest.cpp
    src/model/downloader.cpp
    src/nlp/sampler.cpp
    src/nlp/tokenizer.cpp
    src/llm/kv_cache.cpp
    src/llm/transformer_session.cpp
    src/api/es_runtime.cpp
)

add_library(es_runtime SHARED ${RUNTIME_SOURCES})
target_link_libraries(es_runtime PRIVATE ${CMAKE_DL_LIBS} pthread)
if(OpenMP_CXX_FOUND)
    target_link_libraries(es_runtime PRIVATE OpenMP::OpenMP_CXX)
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

# Milestone 6 Interactive Chat Tool Executable
add_executable(eswin-chat tools/eswin_chat.cpp)
target_link_libraries(eswin-chat PRIVATE es_runtime ${CMAKE_DL_LIBS} pthread)

enable_testing()
add_test(NAME test_dma_alloc COMMAND test_dma_alloc)
add_test(NAME test_dual_die_concurrency COMMAND test_dual_die_concurrency 20)
add_test(NAME test_manifest COMMAND test_manifest)
add_test(NAME test_tokenizer_sampler COMMAND test_tokenizer_sampler)
add_test(NAME test_transformer_session COMMAND test_transformer_session)
add_test(NAME test_runtime_e2e COMMAND test_runtime_e2e)

