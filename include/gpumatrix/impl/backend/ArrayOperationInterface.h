#ifndef BACKEND_COMPOUND_ASSIGN_INTERFACE_H
#define BACKEND_COMPOUND_ASSIGN_INTERFACE_H


#include <gpumatrix/Functional.h>

namespace gpumatrix
{

	namespace impl
	{
	    #define DECLEAR_SCALAR_ARRAY_OP(OPNAME, TYPE) \
	    void scalar_array_##OPNAME( TYPE *odata, TYPE  alpha, const TYPE *idata,  int size);


	    DECLEAR_SCALAR_ARRAY_OP(add,float)
	    DECLEAR_SCALAR_ARRAY_OP(add,double)
	    DECLEAR_SCALAR_ARRAY_OP(sub,float)
	    DECLEAR_SCALAR_ARRAY_OP(sub,double)
	    DECLEAR_SCALAR_ARRAY_OP(mul,float)
	    DECLEAR_SCALAR_ARRAY_OP(mul,double)
	    DECLEAR_SCALAR_ARRAY_OP(div,float)
	    DECLEAR_SCALAR_ARRAY_OP(div,double)


	    #define DECLEAR_ARRAY_ARRAY_OP(OPNAME, TYPE) \
	    void array_##OPNAME( TYPE *odata, const TYPE  * idata1, const TYPE * idata2,  int size) ;



	    DECLEAR_ARRAY_ARRAY_OP(add,float)
	    DECLEAR_ARRAY_ARRAY_OP(add,double)
	    DECLEAR_ARRAY_ARRAY_OP(sub,float)
	    DECLEAR_ARRAY_ARRAY_OP(sub,double)
	    DECLEAR_ARRAY_ARRAY_OP(mul,float)
	    DECLEAR_ARRAY_ARRAY_OP(mul,double)
	    DECLEAR_ARRAY_ARRAY_OP(div,float)
	    DECLEAR_ARRAY_ARRAY_OP(div,double)
	    DECLEAR_ARRAY_ARRAY_OP(cross_entropy,double)
	    DECLEAR_ARRAY_ARRAY_OP(cross_entropy_diff,double)

	    #define DELEAR_ARRAY_ARRAY_COMPOUND_OP(OPNAME, TYPE) \
	    void array_compound_op( TYPE *odata, const TYPE  * idata, int size,const Fcnl_##OPNAME<TYPE,TYPE> & func);


	    DELEAR_ARRAY_ARRAY_COMPOUND_OP(add_eq, double)
	    DELEAR_ARRAY_ARRAY_COMPOUND_OP(add_eq, float)
	    DELEAR_ARRAY_ARRAY_COMPOUND_OP(sub_eq, double)
	    DELEAR_ARRAY_ARRAY_COMPOUND_OP(sub_eq, float)
	    DELEAR_ARRAY_ARRAY_COMPOUND_OP(mul_eq, double)
	    DELEAR_ARRAY_ARRAY_COMPOUND_OP(mul_eq, float)
	    DELEAR_ARRAY_ARRAY_COMPOUND_OP(div_eq, double)
	    DELEAR_ARRAY_ARRAY_COMPOUND_OP(div_eq, float)

	    #define DELEAR_SCALAR_ARRAY_COMPOUND_OP(OPNAME, TYPE) \
	    void scalar_array_compound_op( TYPE *odata, TYPE  alpha,int size,const Fcnl_##OPNAME<TYPE,TYPE> & func) ;

	    DELEAR_SCALAR_ARRAY_COMPOUND_OP(add_eq, double)
	    DELEAR_SCALAR_ARRAY_COMPOUND_OP(add_eq, float)
	    DELEAR_SCALAR_ARRAY_COMPOUND_OP(sub_eq, double)
	    DELEAR_SCALAR_ARRAY_COMPOUND_OP(sub_eq, float)
	    DELEAR_SCALAR_ARRAY_COMPOUND_OP(mul_eq, double)
	    DELEAR_SCALAR_ARRAY_COMPOUND_OP(mul_eq, float)
	    DELEAR_SCALAR_ARRAY_COMPOUND_OP(div_eq, double)
	    DELEAR_SCALAR_ARRAY_COMPOUND_OP(div_eq, float)

	    #define DELEAR_COLWISE_ARRAY_COMPOUND_OP(OPNAME, TYPE) \
	    void colwise_array_compound_op( TYPE *odata, int row, int col, const TYPE * x ,const Fcnl_colwise_##OPNAME<TYPE,TYPE> & func) ;

	    DELEAR_COLWISE_ARRAY_COMPOUND_OP(add_eq, double)
	    DELEAR_COLWISE_ARRAY_COMPOUND_OP(add_eq, float)

	    #define DELEAR_ROWWISE_ARRAY_COMPOUND_OP(OPNAME, TYPE) \
	    void rowwise_array_compound_op( TYPE *odata, int row, int col, const TYPE * x ,const Fcnl_rowwise_##OPNAME<TYPE,TYPE> & func) ;

	    DELEAR_ROWWISE_ARRAY_COMPOUND_OP(add_eq, double)
	    DELEAR_ROWWISE_ARRAY_COMPOUND_OP(add_eq, float)




	}
    
}

#endif