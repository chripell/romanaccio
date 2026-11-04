cmake_minimum_required(VERSION 3.16)
project(lfm3_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../eswin_runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../eswin_runtime/src
)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eswin_runtime/build)

add_executable(lfm3_v2 lfm3_v2.cpp)
target_compile_options(lfm3_v2 PRIVATE -O3 -ffast-math -funroll-loops -Wall -Wextra)
target_link_libraries(lfm3_v2 PRIVATE es_runtime OpenMP::OpenMP_CXX pthread ${CMAKE_DL_LIBS})

set_target_properties(lfm3_v2 PROPERTIES
    BUILD_RPATH "$ORIGIN:$ORIGIN/../eswin_runtime/build:/tmp"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../eswin_runtime/build:/tmp"
)
