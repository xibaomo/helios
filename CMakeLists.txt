cmake_minimum_required(VERSION 3.10.0)
project(helios VERSION 0.1.0 LANGUAGES C CXX CUDA)

# if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
# endif()
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

set(CUDA_DIR /usr/local/cuda-12.9)
include_directories(${CUDA_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/basics)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(src)

install(TARGETS helios
    DESTINATION bin)
install(TARGETS run_utest
    DESTINATION bin)