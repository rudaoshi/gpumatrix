cmake_minimum_required(VERSION 2.8)


project(GPUMatrix)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

add_subdirectory(src)

add_subdirectory(test)






