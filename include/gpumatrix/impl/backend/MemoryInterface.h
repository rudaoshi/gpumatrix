#ifndef MEMORY_INTERFACE_H
#define MEMORY_INTERFACE_H


#include <cstddef>
namespace gpumatrix
{
    namespace impl
    {


		  void alloc_notify();

		  void free_notify();

		  int memory_check();

		  template <typename T>
		  T * alloc(std::size_t size);

		  template <typename T>
		  void free(T * data);

		  template <typename T>
		  void set(T * device_data, const T* host_data, std::size_t size);

		  template <typename T>
		  void get(T * host_data, const T* device_data, std::size_t size);

		  template <typename T>
		  void copy(T * device_dest, const T* device_source, std::size_t size);

		  template <typename T>
		  void zero(T * device_data, std::size_t size);
	  
    }
}



#endif
