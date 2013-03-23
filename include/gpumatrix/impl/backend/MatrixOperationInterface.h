#ifndef MATRIX_OPERATION_INTERFACE_H
#define MATRIX_OPERATION_INTERFACE_H


namespace gpumatrix
{
	namespace impl
	{

			template <typename T> void transpose( T *odata, const T *idata,  int r, int c) ;
		
	}
}


#endif