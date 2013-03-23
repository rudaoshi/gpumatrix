#ifndef BACKEND_FUNCTION_INTERFACE_H
#define BACKEND_FUNCTION_INTERFACE_H


#include <gpumatrix/Functional.h>

namespace gpumatrix
{
    namespace impl
    {

		    
		    template<typename T> T sum(const T * data, int size);

		    template<typename T> T max_element(const T * data, int size);

		    template<typename T> T min_element(const T * data, int size);
		    
		    
		    #define DECLEAR_UNARY_ARRAY_FUNC(OPNAME, TYPE) \
		    void unary_array_op( TYPE *odata, const TYPE  * idata, int size,const Fcnl_##OPNAME<TYPE> & func) ;

		    DECLEAR_UNARY_ARRAY_FUNC(exp, double)
		    DECLEAR_UNARY_ARRAY_FUNC(exp, float)
		    DECLEAR_UNARY_ARRAY_FUNC(neg, double)
		    DECLEAR_UNARY_ARRAY_FUNC(neg, float)
		    DECLEAR_UNARY_ARRAY_FUNC(arrayinv, double)
		    DECLEAR_UNARY_ARRAY_FUNC(arrayinv, float)
		    DECLEAR_UNARY_ARRAY_FUNC(logistic, double)
		    DECLEAR_UNARY_ARRAY_FUNC(logistic, float)
		    DECLEAR_UNARY_ARRAY_FUNC(log, double)
		    DECLEAR_UNARY_ARRAY_FUNC(log, float)
		    
		    
		    template <typename T> void rowwise_sum(T * odata, const T * idata, int r, int c);
		    template <typename T> void colwise_sum(T * odata, const T * idata, int r, int c);

		    #define DECLEAR_BINARY_ARRAY_FUNC(OPNAME, TYPE) \
		    void array_##OPNAME( TYPE *odata, const TYPE  * idata1, const TYPE * idata2,  int size) ;

		    DECLEAR_BINARY_ARRAY_FUNC(cross_entropy,double)
		    DECLEAR_BINARY_ARRAY_FUNC(cross_entropy_diff,double)
	
    }
}


#endif