cmake_minimum_required(VERSION 3.16)
project(eswin_runtime VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

# Core Runtime Library
set(RUNTIME_SOURCES
    src/backend/driver_wrapper.cpp
    src/backend/dma_buffer.cpp
    src/backend/worker_pool.cpp
    src/api/es_runtime.cpp
)

add_library(es_runtime SHARED ${RUNTIME_SOURCES})
target_link_libraries(es_runtime PRIVATE ${CMAKE_DL_LIBS} pthread)
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

enable_testing()
add_test(NAME test_dma_alloc COMMAND test_dma_alloc)
add_test(NAME test_dual_die_concurrency COMMAND test_dual_die_concurrency 100)
