#ifndef FUNCTION_INTERFACE_H
#define FUNCTION_INTERFACE_H

namespace gpumatrix
{
    namespace impl
    {
		template <typename E>
		typename E::value_type squaredNorm(const E & m);
		
		template <typename E>
		typename E::value_type sum(const E & m);
		
		template <typename E>
		typename E::value_type min(const E & m);
		
		template <typename E>
		typename E::value_type max(const E & m);
		
		template <typename E1, typename E2>
		typename E1::value_type dot(const E1 & v1, const E2 & v2);
		
    }
}


#endif