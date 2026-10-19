cmake_minimum_required(VERSION 3.16)
project(qwen2_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eswin_runtime/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eswin_runtime/build)

add_executable(qwen2_v2 qwen2_v2.cpp)
target_link_libraries(qwen2_v2 PRIVATE es_runtime pthread ${CMAKE_DL_LIBS})

set_target_properties(qwen2_v2 PROPERTIES
    BUILD_RPATH "$ORIGIN/../eswin_runtime/build:$ORIGIN"
    INSTALL_RPATH "$ORIGIN/../eswin_runtime/build:$ORIGIN"
)
