INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)


set(srcfiles 
    ./impl/backend/cuda/ArrayOperationImpl.cu
    ./impl/backend/cuda/MatrixOperationImpl.cu
    ./impl/backend/cuda/BlasImpl.cpp
    ./impl/backend/cuda/FunctionImpl.cu
    ./impl/backend/cuda/MemoryImpl.cpp
)

#Include FindCUDA script
INCLUDE(FindCUDA)

#Rule to build executable program matrixMult from matrixmul.cu
# and matrixMul_gold.cpp
CUDA_ADD_LIBRARY(GPUMatrix SHARED ${srcfiles})

