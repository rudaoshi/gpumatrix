
#include <gpumatrix/CORE>



// TestLeastSquareSVM.cpp : 定义控制台应用程序的入口点。
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


	struct MapOperationData
	{

		MapOperationData()
		{
			cublasInit();
		}

		~MapOperationData()
		{ 
			cublasShutdown();
		}
	};

	typedef test_group<MapOperationData> tg;
	typedef tg::object object;
	tg MapOperationTestGroup("MapOperationTest");


	// Test Map
	template<>
	template<>
	void object::test<1>()
	{
		Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(20,60);
		Eigen::MatrixXd h_A2 = Eigen::MatrixXd::Random(20,60);
		Eigen::VectorXd h_B = Eigen::VectorXd::Random(30);
		Eigen::VectorXd h_C = Eigen::VectorXd::Random(100);

		Matrix<double> d_A(h_A),d_A2(h_A2);
		Vector<double> d_B(h_B),d_C(h_C);



		Vector<double> d_D;

		Map<Matrix<double>> d_mA(d_A.data()+20,20,30);
		Map<Matrix<double>> d_mA2(d_A2.data()+40,20,30);

		Matrix<double> d_E = d_mA + d_mA2;
		Matrix<double> d_F = d_mA - d_mA2;
		Matrix<double> d_G = d_mA* d_mA2.transpose() ;

		Map<Vector<double>> d_vC(d_C.data()+20,20);
		
		d_D =  d_mA.transpose()*d_vC + d_B;

		Eigen::Map<Eigen::MatrixXd> h_mA(h_A.data()+20,20,30);
		Eigen::Map<Eigen::MatrixXd> h_mA2(h_A2.data()+40,20,30);

		Eigen::MatrixXd h_E = h_mA + h_mA2;
		Eigen::MatrixXd h_F = h_mA - h_mA2;
		Eigen::MatrixXd h_G = h_mA * h_mA2.transpose();

		Eigen::Map<Eigen::VectorXd> h_vC(h_C.data()+20,20);

		Eigen::VectorXd h_D = h_mA.transpose()*h_vC + h_B;
		
		double error1 = (h_D - (Eigen::VectorXd)d_D).squaredNorm();
		double error2 = (h_E - (Eigen::MatrixXd)d_E).squaredNorm();
		double error3 = (h_F - (Eigen::MatrixXd)d_F).squaredNorm();
		double error4 = (h_G - (Eigen::MatrixXd)d_G).squaredNorm();

		ensure(error1 < 1e-5);
		ensure(error2 < 1e-5);
		ensure(error3 < 1e-5);
		ensure(error4 < 1e-5);

		
	}


}