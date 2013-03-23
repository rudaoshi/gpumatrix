#include <gpumatrix/impl/backend/MemoryInterface.h>

//#include <boost/thread.hpp>


namespace gpumatrix
{
	namespace impl
	{
		int memory_counter = 0;

		//boost::mutex memory_counter_update_mutex;

		void alloc_notify()
		{
			//{
			//	boost::mutex::scoped_lock lock(memory_counter_update_mutex);
			//	memory_counter ++;
			//}
		}


		void free_notify()
		{
			//{
			//	boost::mutex::scoped_lock lock(memory_counter_update_mutex);
			//	memory_counter --;
			//}
		}

		int memory_check()
		{
			//{
			//	boost::mutex::scoped_lock lock(memory_counter_update_mutex);
			//	return memory_counter;
			//}

			return 0;
		}

	}
}