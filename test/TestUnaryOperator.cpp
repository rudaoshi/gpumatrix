

#include <gpumatrix/CORE>



// TestLeastSquareSVM.cpp : �������̨Ӧ�ó������ڵ㡣
//
#include <tut/tut.hpp>
#include <stdexcept>
#include <ctime>
#include <iostream>
#include "Util.h"

using std::runtime_error;
using namespace std;

/**
* This example test group tests shared_ptr implementation
* as tutorial example for TUT framework.
*/
namespace tut
{
	using namespace gpumatrix;


	struct UnaryOperationData
	{

		UnaryOperationData()
		{
			cublasInit();
		}

		~UnaryOperationData()
		{ 
			cublasShutdown();
		}
	};

	typedef test_group<UnaryOperationData> tg;
	typedef tg::object object;
	tg UnaryOperationTestGroup("UnaryOperationTest");

	// Test squaredNorm
	template<>
	template<>
	void object::test<1>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;



			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row,col);
			Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(row,col);

			Matrix<double> d_A(h_A), d_B(h_B);

			double norm1 = (d_A-d_B).squaredNorm();

			double norm2 = (h_A-h_B).squaredNorm();
			
			double error1 = abs(norm1-norm2);


			ensure("squared norm operation pass", error1 < 1e-5);

		}
	}

		// Test sum and min max
	template<>
	template<>
	void object::test<2>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;



			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row,col);
			Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(row,col);

			Matrix<double> d_A(h_A), d_B(h_B);

			double sum1 = d_A.sum();

			double sum2 = h_A.sum();
			
			double error1 = abs(sum1-sum2);

			double min1 = d_A.minCoeff();

			double min2 = h_A.minCoeff();

			double error2 = abs(min1-min2);

			double max1 = d_A.maxCoeff();

			double max2 = h_A.maxCoeff();

			double error3 = abs(max1-max2);

			ensure("sum operation pass", error1 < 1e-5);
			ensure("min operation pass", error2 < 1e-5);
			ensure("max operation pass", error3 < 1e-5);

		}
	}

	// Test dot
	template<>
	template<>
	void object::test<3>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;

			Eigen::VectorXd h_A = Eigen::VectorXd::Random(row);
			Eigen::VectorXd h_B = Eigen::VectorXd::Random(row);

			Vector<double> d_A(h_A), d_B(h_B);

			double dot1 = d_A.dot(d_B);

			double dot2 = h_A.dot(h_B);

			double error1 = abs(dot1-dot2);

			ensure("dot operation pass", error1 < 1e-5);

		}
	}
}