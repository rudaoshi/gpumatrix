# CmakeLists.txt in Test dir
# Make sure the compiler can find include files from our Hello library.
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include /usr/local/include/eigen3)
# Add binary called "helloWorld" that is built from the source file "test.cc".
# The extension is automatically found.

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++0x")


set(srcfiles 
    main.cpp
    TestArrayOperation.cpp
    TestGPUMatrix.cpp
    TestMapOperation.cpp
    TestMatrixAlgebra.cpp
    TestMatrixVectorAlgebra.cpp
    TestUnaryOperator.cpp
    TestVectorAlgebra.cpp
)

#Include FindCUDA script
INCLUDE(FindCUDA)

#Search CUDA.
FIND_PACKAGE(CUDA)


include_directories(${CUDA_INCLUDE_DIRS})
set(LIBS ${LIBS} ${CUDA_LIBRARIES})



CUDA_ADD_EXECUTABLE(testGPUMatrix ${srcfiles})
# Link the executable to the learning library.
TARGET_LINK_LIBRARIES(testGPUMatrix GPUMatrix ${LIBS} )
CUDA_ADD_CUBLAS_TO_TARGET( testGPUMatrix )


