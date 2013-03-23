

#include <gpumatrix/CORE>



// TestLeastSquareSVM.cpp : 定义控制台应用程序的入口点。
//
#include <tut/tut.hpp>
#include <stdexcept>
#include <ctime>
#include <iostream>
#include "Util.h"

#include <Eigen/Core>

using std::runtime_error;
using namespace std;

/**
* This example test group tests shared_ptr implementation
* as tutorial example for TUT framework.
*/
namespace tut
{
	using namespace gpumatrix;


	struct ArrayOperationData
	{

		ArrayOperationData()
		{
			cublasInit();
		}

		~ArrayOperationData()
		{ 
			cublasShutdown();
		}
	};

	typedef test_group<ArrayOperationData> tg;
	typedef tg::object object;
	tg ArrayOperationTestGroup("ArrayOperationTest");


	// Test Array
	template<>
	template<>
	void object::test<1>()
	{
		Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(20,50);
		Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(20,50);
		Eigen::MatrixXd h_C = Eigen::MatrixXd::Random(50,40);
		
		Matrix<double> d_A(h_A);
		Matrix<double> d_B(h_B);

		Matrix<double> d_D;

		d_D = (d_A.array()*d_B.array()).matrix();

		Eigen::MatrixXd h_D = (h_A.array()*h_B.array()).matrix();
		
		
		double error1 = (h_D - (Eigen::MatrixXd)d_D).squaredNorm();


		ensure(error1 < 1e-5);


		
	}

		// Test Logistic
	template<>
	template<>
	void object::test<2>()
	{

		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;
			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row,col), h_B;
		
			Matrix<double> d_A(h_A), d_B;

			d_B = d_A.array().logistic();

			h_B = (1 + (-h_A).array().exp()).inverse();

			ensure(check_diff(h_B,d_B));

		}
		
	}



	
	// Test Mixed Matrix Array Operation
	template<>
	template<>
	void object::test<3>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;
			int col2 = rand()%1000+1;

			Eigen::MatrixXd W = Eigen::MatrixXd::Random(row,col);
			Eigen::MatrixXd Delta = Eigen::MatrixXd::Random(row,col2);
			Eigen::MatrixXd Input = Eigen::MatrixXd::Random(col,col2);

			gpumatrix::Matrix<double> gDelta = Delta, gInput = Input, gW = W, gResult;

			gResult = ((gW.transpose()*gDelta).array()*gInput.array())*(1 - gInput.array());

			Eigen::MatrixXd Result = ((W.transpose()*Delta).array()*Input.array())*(1 - Input.array());

			ensure(check_diff(Result,gResult));

		}
	}

		// Test Mixed Matrix Array Operation
	template<>
	template<>
	void object::test<4>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;
			int col2 = rand()%1000+1;

			Eigen::MatrixXd W = Eigen::MatrixXd::Random(row,col);
			Eigen::MatrixXd R = Eigen::MatrixXd::Random(row,col);

			gpumatrix::Matrix<double> gR = R, gW = W, gResult;

			(2*gW.array()).eval();



		}
	}

}