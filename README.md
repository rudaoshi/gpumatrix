# gpumatrix


A matrix and array library on GPU with interface compatible with Eigen.

## Features

* Supports CUDA back-end.
* Most common Array and Matrix operations are supported. See test suite for more details.
* Implemented interfaces are compatible with Eigen 3. Program using Eigen is easy to port to GPU using GPUMatrix.



## Missing but planned

* All vectors are column vectors. Row vectors is not implemented;
* Multiple back-end support, including OpenCL and MKL;
* Interface for user defined member functions, like Eigen;
* Many functions and operations are waiting to be implemented.

## Build and test

* Build as a standard cmake project;
* To correctly build the test, Eigen3 is needed. It's include-path can be specified by EIGEN3_INCLUDE_DIR variable. 

## Thanks

GPUMatrix adopts the template expression architecture of TVMet library from tvmet.sourceforge.net. I'd like to appreciate Olaf Petzold for his great work on TVMet library.


The project is previous hosted at Google Code: http://code.google.com/p/gpumatrix/.
