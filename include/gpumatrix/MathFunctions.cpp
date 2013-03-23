#include "MathFunctions.h"




#include <gpumatrix/backend/cuda/Function.h>
#include <gpumatrix/backend/cuda/array_kernel.h>
namespace gpumatrix
{
	double cross_entropy(const Matrix<double> & X, const Matrix<double> & R)
	{
		Matrix<double> result(X.rows(),X.cols());

		impl::array_cross_entropy(result.data(),X.data(),R.data(),X.size());

		return -1.0/double(X.cols())*result.sum();
	}


	Matrix<double> cross_entropy_delta(const Matrix<double> & X, const Matrix<double> & R)
	{
		Matrix<double> result(X.rows(),X.cols());

		impl::array_cross_entropy_diff(result.data(),X.data(),R.data(),X.size());

		return 1.0/double(X.cols())*result;
	}

}