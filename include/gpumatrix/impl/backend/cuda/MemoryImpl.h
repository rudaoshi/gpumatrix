#ifndef GPU_MEMORY_H
#define GPU_MEMORY_H


#include <gpumatrix/impl/backend/MemoryInterface.h>
#include <cuda.h>
#include <cublas.h>
#include <cstddef> 
#include <stdexcept>

namespace gpumatrix
{
	namespace impl
	{
		void alloc_notify();

		void free_notify();

		int memory_check();

		template <typename T>
		T * alloc(std::size_t size)
		{
			T * data;
			cublasStatus status = cublasAlloc (size, sizeof(T),(void **) &data);
			if (status != CUBLAS_STATUS_SUCCESS)
				throw std::runtime_error("GPU Memory Allocation Failed");

			alloc_notify();

			return data;
		}


		template <typename T>
		void free(T * data)
		{
			if ( data == 0)
				return;

			cublasStatus status = cublasFree (data);
			if (status != CUBLAS_STATUS_SUCCESS)
				throw std::runtime_error("GPU Memory Free Failed");

			free_notify();
		}

		template <typename T>
		void set(T * device_data, const T* host_data, std::size_t size)
		{
			cublasStatus status = cublasSetVector(size,sizeof(T),host_data,1,device_data,1);
			if (status != CUBLAS_STATUS_SUCCESS)
				throw std::runtime_error("GPU Memory SetVector Failed");
		}

		template <typename T>
		void get(T * host_data, const T* device_data, std::size_t size)
		{
			cublasStatus status = cublasGetVector (size, sizeof(T), device_data,1, host_data, 1);
			if (status != CUBLAS_STATUS_SUCCESS)
				throw std::runtime_error("GPU Memory GetVector Failed");
		}

		template <typename T>
		void copy(T * device_dest, const T* device_source, std::size_t size)
		{
			cudaError_t cudaError = cudaMemcpy(device_dest, device_source,size*sizeof(T), cudaMemcpyDeviceToDevice);
			
			if (cudaError != cudaSuccess)
				throw std::runtime_error(cudaGetErrorString(cudaError));
		}

		template <typename T>
		void zero(T * device_data, std::size_t size)
		{
			cudaError_t cudaError = cudaMemset(device_data, 0,size*sizeof(T));
			
			if (cudaError != cudaSuccess)
				throw std::runtime_error(cudaGetErrorString(cudaError));
		}
		
	}
}

#endif