#ifndef FUNCTION_IMPL_H
#define FUNCTION_IMPL_H

#include <gpumatrix/impl/backend/Interface.h>



namespace gpumatrix
{

	namespace impl
	{

		template <typename E>
		typename E::value_type squaredNorm(const E & m)
		{
			typename E::value_type norm =  impl::nrm2(m.size(),m.data(),1) ;
			return norm*norm;
		}



		template <typename E>
		typename E::value_type sum(const E & m)
		{
			return impl::sum(m.data(),m.size());
			
		}

		template <typename E>
		typename E::value_type min(const E & m)
		{
			return impl::min_element(m.data(),m.size());
			
		}

		template <typename E>
		typename E::value_type max(const E & m)
		{
			return impl::max_element(m.data(),m.size());
			
		}

		template <typename E1, typename E2>
		typename E1::value_type dot(const E1 & v1, const E2 & v2)
		{
			return impl::dot(v1.size(),v1.data(),1,v2.data(),1) ;
		}


	}
}
#endif