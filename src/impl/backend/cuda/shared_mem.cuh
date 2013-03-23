#ifndef SHARED_MEM_CU_H
#define SHARED_MEM_CU_H


namespace gpumatrix
{
	namespace impl
	{

			template <class T>
			class SharedMem
			{
			public:
				// Ensure that we won't compile any un-specialized types
				__device__  T* getPointer() { return 0; };
			};

			// specialization for int
			template <>
			class SharedMem <double>
			{
			public:
				__device__ double* getPointer() { extern __shared__ double s_double[]; return s_double; }
			};

			// specialization for float
			template <>
			class SharedMem <float>
			{
			public:
				__device__ float* getPointer() { extern __shared__ float s_float[]; return s_float; }
			};
		
	}

}

#endif