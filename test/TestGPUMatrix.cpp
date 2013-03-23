// TestGPUMatrix.cpp : 定义控制台应用程序的入口点。
//

#include <gpumatrix/Matrix.h>
#include <gpumatrix/Vector.h>
#include <gpumatrix/Array.h>



// TestLeastSquareSVM.cpp : 定义控制台应用程序的入口点。
//
#include <tut/tut.hpp>
#include <stdexcept>
#include <ctime>
#include <iostream>

using std::runtime_error;
using namespace std;

/**
* This example test group tests shared_ptr implementation
* as tutorial example for TUT framework.
*/
namespace tut
{
	using namespace gpumatrix;


	struct GPUMatrixData
	{

		GPUMatrixData()
		{
			std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
			cublasInit();
		}

		~GPUMatrixData()
		{ 
			cublasShutdown();
		}
	};

	typedef test_group<GPUMatrixData> tg;
	typedef tg::object object;
	tg GPUMatrixGroup("GPUMatrixData");

	// =================================================
	// Constructors
	// =================================================


	// Test Matrix Matrix multiplication
	template<>
	template<>
	void object::test<1>()
	{
		try
		{
			clock_t cpu_total = 0;
			clock_t gpu_total = 0;

			for (int i = 0;i<1;i++)
			{
				int row = rand()%1000+1;
				int col = rand()%1000+1;
				int col2 = rand()%1000+1;


				Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(col,row);
				Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(col,col2);
				Matrix<double> d_A(h_A);
				Matrix<double> d_B(h_B);

				Matrix<double> d_C;
		
				cudaThreadSynchronize();
				clock_t gpu_start = clock();
				d_C =  d_A.transpose()*d_B;
				cudaThreadSynchronize();
				clock_t gpu_end = clock();

				gpu_total += (gpu_end - gpu_start);
		
				Eigen::MatrixXd h_C;
				
				clock_t cpu_start = clock();
				h_C = h_A.transpose()*h_B;
				clock_t cpu_end = clock();

				cpu_total += (cpu_end - cpu_start);

				double error1 = (h_C - (Eigen::MatrixXd)d_C).squaredNorm();
				ensure(error1 < 1e-5);

				//Eigen::MatrixXd h_S = Eigen::MatrixXd::Random(50,50);

				//Matrix<double> d_S(h_S);

				//d_S = d_S*d_S;
				//h_S = h_S*h_S;

				//double error2 = (h_S - (Eigen::MatrixXd)d_S).squaredNorm();
				//ensure(error2 < 1e-5);

				
			}
		
			std::cout << "gpu operation costing time of " << gpu_total << " while cpu operation costing time of " << cpu_total << std::endl;
		}
		catch (std::exception & e)
		{
			std::cout<<"=============="<<e.what() <<"================" << std::endl;
		}
		std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	}

	// Test Scalar Matrix multiplication
	template<>
	template<>
	void object::test<2>()
	{
		Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(20,20);
		Matrix<double> d_A(h_A);

		Matrix<double> d_B,d_C;
		
		d_B =  2*d_A;
		d_C =  d_A * 2;                                                         

		Eigen::MatrixXd h_B = 2*h_A;

		double error1 = (h_B - (Eigen::MatrixXd)d_B).squaredNorm();
		double error2 = (h_B - (Eigen::MatrixXd)d_C).squaredNorm();
		
		ensure(error1 < 1e-5);
		ensure(error2 < 1e-5);
		
	}

	// Test Matrix Matrix plas and Sub
	template<>
	template<>
	void object::test<3>()
	{
		Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(20,20);
		Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(20,20);

		Matrix<double> d_A(h_A);
		Matrix<double> d_B(h_B);

		Matrix<double> d_C,d_D;
		
		d_C =  d_A + d_B;
		d_D =  d_A - d_B;

		Eigen::MatrixXd h_C = h_A + h_B;
		Eigen::MatrixXd h_D = h_A - h_B;

		double error1 = (h_C - (Eigen::MatrixXd)d_C).squaredNorm();
		double error2 = (h_D - (Eigen::MatrixXd)d_D).squaredNorm();

		ensure(error1 < 1e-5);
		ensure(error2 < 1e-5);
		
	}



	// Test Matrix Transpose
	template<>
	template<>
	void object::test<4>()
	{
		Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(2,5);
		Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(2,5);
		Eigen::MatrixXd h_B2 = Eigen::MatrixXd::Random(5,2);

		Matrix<double> d_A(h_A);
		Matrix<double> d_B(h_B);
		Matrix<double> d_B2(h_B2);

		Matrix<double> d_C,d_E,d_F,d_G;

		Eigen::MatrixXd h_C,h_D,h_E,h_F,h_G;
		
		d_C = d_A.transpose();
		h_C = h_A.transpose();

		Eigen::MatrixXd h_C2 = (Eigen::MatrixXd)d_C;
		double error0 = (h_C - h_C2).squaredNorm();

		ensure(error0 < 1e-5);

		d_C =  d_A.transpose()*d_B;
		Matrix<double>  d_D = d_A*d_B.transpose();
		d_E = d_A.transpose()*d_B2.transpose();
		d_F = d_B2.transpose().transpose();
		d_G = (d_A*d_B2).transpose();

		
		h_C = (h_A.transpose()*h_B);
		h_D = h_A*h_B.transpose();
		h_E = h_A.transpose()*h_B2.transpose();
		h_F = h_B2.transpose().transpose();
		h_G = (h_A*h_B2).transpose();

		double error1 = (h_C - (Eigen::MatrixXd)d_C).squaredNorm();
		double error2 = (h_D - (Eigen::MatrixXd)d_D).squaredNorm();
		double error3 = (h_E - (Eigen::MatrixXd)d_E).squaredNorm();
		double error4 = (h_F - (Eigen::MatrixXd)d_F).squaredNorm();
		double error5 = (h_G - (Eigen::MatrixXd)d_G).squaredNorm();

		ensure(error1 < 1e-5);
		ensure(error2 < 1e-5);
		ensure(error3 < 1e-5);
		ensure(error4 < 1e-5);
		ensure(error5 < 1e-5);

	}

	// Test Vector Vector algorithm
	template<>
	template<>
	void object::test<5>()
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

		// Test Matrix Vector Multiplication
	template<>
	template<>
	void object::test<6>()
	{
		Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(20,50);
		Eigen::VectorXd h_B = Eigen::VectorXd::Random(50);

		Matrix<double> d_A(h_A);
		Vector<double> d_B(h_B);

		Vector<double> d_C;
		
		d_C =  d_A*d_B;


		Eigen::VectorXd h_C = h_A*h_B;


		double error1 = (h_C - (Eigen::VectorXd)d_C).squaredNorm();

		ensure(error1 < 1e-5);

		
	}

		// Test Complex algorithm
	template<>
	template<>
	void object::test<7>()
	{
		Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(20,50);
		Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(30,50);
		Eigen::MatrixXd h_A1 = Eigen::MatrixXd::Random(20,50);
		Eigen::MatrixXd h_B1 = Eigen::MatrixXd::Random(30,50);
		Eigen::MatrixXd h_C = Eigen::MatrixXd::Random(20,30);
		Eigen::MatrixXd h_D = Eigen::MatrixXd::Random(20,30);;

		Matrix<double> d_A(h_A);
		Matrix<double> d_B(h_B);
		Matrix<double> d_A1(h_A1);
		Matrix<double> d_B1(h_B1);
		Matrix<double> d_C(h_C);
		Matrix<double> d_D(h_D);
		
		
		d_D = 0.5*d_D + 0.01*( (d_A*d_B.transpose()-d_A1*d_B1.transpose())/100 - 0.9*d_C);
		
		h_D = 0.5*h_D + 0.01*( (h_A*h_B.transpose()-h_A1*h_B1.transpose())/100 - 0.9*h_C);




		double error1 = (h_D - (Eigen::MatrixXd)d_D).squaredNorm();

		ensure(error1 < 1e-5);

		
	}


	// Test Map
	template<>
	template<>
	void object::test<8>()
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

	// Test Array
	template<>
	template<>
	void object::test<9>()
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

	// Test unary operator
	template<>
	template<>
	void object::test<10>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;



			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(col,row);

			Matrix<double> d_A(h_A);

			Vector<double> d_B = d_A.rowwise().sum();

			Eigen::VectorXd h_B = h_A.rowwise().sum();

			double error1 = (h_B - (Eigen::VectorXd)d_B).squaredNorm();


			ensure(error1 < 1e-5);

		}
	}

	// Test rowwise_sum
	template<>
	template<>
	void object::test<11>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;



			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(col,row);
			Eigen::MatrixXd h_B = Eigen::MatrixXd::Random(col,row);

			Matrix<double> d_A(h_A), d_B(h_B);

			double norm1 = (d_A-d_B).squaredNorm();

			double norm2 = (h_A-h_B).squaredNorm();
			
			double error1 = abs(norm1-norm2);


			ensure(error1 < 1e-5);

		}
	}

}



