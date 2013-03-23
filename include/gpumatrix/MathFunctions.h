#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <gpumatrix/Matrix.h>



namespace gpumatrix
{

	double cross_entropy(const Matrix<double> & X, const Matrix<double> & R);

	Matrix<double> cross_entropy_delta(const Matrix<double> & X, const Matrix<double> & R);

}




#endif