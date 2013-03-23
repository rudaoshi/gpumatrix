#include <gpumatrix/CORE>



// TestLeastSquareSVM.cpp : ��������̨Ӧ�ó��������ڵ㡣
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


	struct VectorAlgebraData
	{

		VectorAlgebraData()
		{
			cublasInit();
		}

		~VectorAlgebraData()
		{ 
			cublasShutdown();

			int mem_check = gpumatrix::impl::memory_check();

			if (mem_check != 0 )
				throw std::runtime_error("Memory management in GPUMatrix error!");
		}
	};

	typedef test_group<VectorAlgebraData> tg;
	typedef tg::object object;
	tg VectorAlgebraTestGroup("VectorAlgebraTest");


	// Test Vector linear operation
	template<>
	template<>
	void object::test<1>()
	{
		Eigen::VectorXd h_A = Eigen::VectorXd::Random(20);
		Eigen::VectorXd h_B = Eigen::VectorXd::Random(20);

		double alpha = 2.0;

		Vector<double> d_A(h_A);
		Vector<double> d_B(h_B);

		Vector<double> d_C,d_D,d_E,d_F;
		
		d_C =  d_A + d_B;
		d_D =  d_A - d_B;
		d_E =  alpha * d_A;
		d_F =  d_A * alpha;

		Eigen::VectorXd h_C, h_D,h_E, h_F;
		h_C = h_A + h_B;
		h_D = h_A - h_B;
		h_E =  alpha * h_A;
		h_F =  h_A * alpha;

		double error1 = (h_C - (Eigen::VectorXd)d_C).squaredNorm();
		double error2 = (h_D - (Eigen::VectorXd)d_D).squaredNorm();
		double error3 = (h_E - (Eigen::VectorXd)d_E).squaredNorm();
		double error4 = (h_F - (Eigen::VectorXd)d_F).squaredNorm();

		ensure(error1 < 1e-5);
		ensure(error2 < 1e-5);
		ensure(error3 < 1e-5);
		ensure(error4 < 1e-5);
		
	}


}