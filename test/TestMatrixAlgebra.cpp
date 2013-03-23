

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


	struct MatrixAlgebraData
	{

		MatrixAlgebraData()
		{
			cublasInit();
		}

		~MatrixAlgebraData()
		{ 
			cublasShutdown();

			int mem_check = gpumatrix::impl::memory_check();

			if (mem_check != 0 )
				throw std::runtime_error("Memory management in GPUMatrix error!");
		}
	};

	typedef test_group<MatrixAlgebraData> tg;
	typedef tg::object object;
	tg MatrixAlgebraTestGroup("MatrixAlgebraTest");


	// Matrix Matrix Add Sub
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
			Eigen::MatrixXd h_C = Eigen::MatrixXd::Random(row,col);

			Matrix<double> d_A(h_A), d_B(h_B), d_C(h_C);

			Matrix<double> d_D,d_E;
		
			d_D =  d_A + d_B;
			d_E =  d_A - d_B;

			Matrix<double> d_F = d_A + d_B - d_C;
			Matrix<double> d_G = - d_A + d_B + d_C;

			Eigen::MatrixXd h_D = h_A + h_B;
			Eigen::MatrixXd h_E = h_A - h_B;

			Eigen::MatrixXd h_F = h_A + h_B - h_C;
			Eigen::MatrixXd h_G = - h_A + h_B + h_C;

			ensure(check_diff(h_D,d_D));
			ensure(check_diff(h_E,d_E));
			ensure(check_diff(h_F,d_F));
			ensure(check_diff(h_G,d_G));
		}
		
	}

	// Test Scalar Matrix operation
	template<>
	template<>
	void object::test<2>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;
		
			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row,col);
			Matrix<double> d_A(h_A);

			Matrix<double> d_B,d_C,d_D;
		
			d_B =  2*d_A;
			d_C =  d_A * 2;     
			d_D = d_A / 2;

			Eigen::MatrixXd h_B = 2*h_A;
			Eigen::MatrixXd h_C = h_A*2;
			Eigen::MatrixXd h_D = h_A / 2;

			ensure(check_diff(h_B,d_B));
			ensure(check_diff(h_C,d_C));
			ensure(check_diff(h_D,d_D));
		}
		
	}

	// Test Matrix Transpose
	template<>
	template<>
	void object::test<3>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;
		
			Eigen::MatrixXd hA = Eigen::MatrixXd::Random(row,col);

			Matrix<double> dA(hA);

			Matrix<double> dB,dC,dD,dE;

			dB = dA.transpose();
			dC = dB.transpose();
			dD = dA.transpose().transpose();
			dE = dD.transpose().transpose().transpose();

			Eigen::MatrixXd hB,hC,hD,hE;
		
			hB = hA.transpose();
			hC = hB.transpose();
			hD = hA.transpose().transpose();
			hE = hD.transpose().transpose().transpose();
			
			ensure(check_diff(hB,dB));
			ensure(check_diff(hC,dC));
			ensure(check_diff(hD,dD));
			ensure(check_diff(hE,dE));
			

		}

	}
	
	// Test Matrix Matrix multiplication
	template<>
	template<>
	void object::test<4>()
	{
		try
		{
			for (int i = 0;i<10;i++)
			{
				int row = rand()%1000+1;
				int col = rand()%1000+1;
				int col2 = rand()%1000+1;

				Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(col,row);
				Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(row,col2);
				Eigen::MatrixXd h_C = Eigen::MatrixXd::Random(col2,col);

				Matrix<double> d_A(h_A),d_B(h_B),d_C(h_C);
				Matrix<double> d_D,d_E,d_F,d_G,d_H,d_I;
		
				d_D = d_A * d_B;
				d_E = d_B * d_C;
				d_F = d_A * d_B * d_C;
				//d_G = d_A * d_B * d_C * d_A * d_B * d_C;
				d_H = d_F * d_F;
				d_I = d_F; d_I = d_I * d_I;

				Eigen::MatrixXd h_D,h_E,h_F,h_G,h_H,h_I;
				
				h_D = h_A * h_B;
				h_E = h_B * h_C;
				h_F = h_A * h_B * h_C;
				//h_G = h_A * h_B * h_C * h_A * h_B * h_C;
				h_H = h_F * h_F;
				h_I = h_F; h_I = h_I * h_I;


				ensure(check_diff(h_D,d_D));
				ensure(check_diff(h_E,d_E));
				ensure(check_diff(h_F,d_F));
				//ensure(check_diff(h_G,d_G));
				ensure(check_diff(h_H,d_H));
				ensure(check_diff(h_I,d_I));
				
			}
		
		}
		catch (std::exception & e)
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
			std::cout<<"=============="<<e.what() <<"================" << std::endl;
			throw e;
		}

	}

	// Test Matrix Transpose Matrix Mutiplication
	template<>
	template<>
	void object::test<5>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = 784;
			int col = 5003;

			Eigen::MatrixXd hA = Eigen::MatrixXd::Random(row,col);
			Eigen::MatrixXd hB = Eigen::MatrixXd::Random(row,col);
			Eigen::MatrixXd hB2 = Eigen::MatrixXd::Random(col,row);
			Eigen::MatrixXd hC, hD, hE, hF;

			Matrix<double> dA(hA), dB(hB), dB2(hB2), dC, dD, dE, dF;
			
			dC = dA.transpose()*dB;
			dD = dA*dB.transpose();
			dE = dA.transpose()*dB2.transpose();
			dF = (dA*dB2).transpose();

			hC = hA.transpose()*hB;
			hD = hA*hB.transpose();
			hE = hA.transpose()*hB2.transpose();
			hF = (hA*hB2).transpose();

			ensure(check_diff(hC,dC));
			ensure(check_diff(hD,dD));
			ensure(check_diff(hE,dE));
			ensure(check_diff(hF,dF));
		}
	}

	// Test General Linear Matrix Operation
	template<>
	template<>
	void object::test<6>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;

			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row,col);
			Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(row,col);
			Eigen::MatrixXd h_C = Eigen::MatrixXd::Random(col,row);

			double alpha = 1.5, beta = 2.8, gamma = -2.1;

			Matrix<double> d_A(h_A), d_B(h_B), d_C(h_C);

			Matrix<double> d_D,d_E;
		
			d_D =  alpha * d_A + gamma * d_B;
			d_E =  alpha * d_A - beta * d_B;

			Matrix<double> d_F = alpha * d_A + beta * d_B - gamma * d_C.transpose();
			Matrix<double> d_G = - alpha * d_A + beta * d_B + gamma * d_C.transpose();

			Eigen::MatrixXd h_D = alpha * h_A + gamma * h_B;
			Eigen::MatrixXd h_E = alpha * h_A - beta * h_B;

			Eigen::MatrixXd h_F = alpha * h_A + beta * h_B - gamma * h_C.transpose();
			Eigen::MatrixXd h_G = - alpha * h_A + beta * h_B + gamma * h_C.transpose();

			ensure(check_diff(h_D,d_D));
			ensure(check_diff(h_E,d_E));
			ensure(check_diff(h_F,d_F));
			ensure(check_diff(h_G,d_G));
		}
	}

	// Test General Complex Operation
	template<>
	template<>
	void object::test<7>()
	{
		for (int i = 0;i<10;i++)
		{
			int row1 = rand()%1000+1;
			int col1 = rand()%1000+1;
			int row2 = rand()%1000+1;
			int col2 = rand()%1000+1;

			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row1,col1);
			Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(row2,col1);
			Eigen::MatrixXd h_A1 = Eigen::MatrixXd::Random(row1,col2);
			Eigen::MatrixXd h_B1 = Eigen::MatrixXd::Random(row2,col2);
			Eigen::MatrixXd h_C = Eigen::MatrixXd::Random(row1,row2);
			Eigen::MatrixXd h_D = Eigen::MatrixXd::Random(row1,row2);;

			Matrix<double> d_A(h_A);
			Matrix<double> d_B(h_B);
			Matrix<double> d_A1(h_A1);
			Matrix<double> d_B1(h_B1);
			Matrix<double> d_C(h_C);
			Matrix<double> d_D(h_D);
		
		
			d_D = 0.5*d_D + 0.01*( (d_A*d_B.transpose()-d_A1*d_B1.transpose())/100 - 0.9*d_C);
			h_D = 0.5*h_D + 0.01*( (h_A*h_B.transpose()-h_A1*h_B1.transpose())/100 - 0.9*h_C);

			ensure(check_diff(h_D,d_D));

		}

		
	}




}



