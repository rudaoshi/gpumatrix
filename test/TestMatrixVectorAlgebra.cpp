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


	struct MatrixVectorAlgebraData
	{

		MatrixVectorAlgebraData()
		{
			cublasInit();
		}

		~MatrixVectorAlgebraData()
		{ 
			cublasShutdown();
		}
	};

	typedef test_group<MatrixVectorAlgebraData> tg;
	typedef tg::object object;
	tg MatrixVectorAlgebraTestGroup("MatrixVectorAlgebraTest");

	
	// Test Matrix Vector Multiplication
	template<>
	template<>
	void object::test<1>()
	{
		for (int i = 0;i<10;i++)
		{
			int row1 = rand()%1000+1;
			int col1 = rand()%1000+1;
		
			int row2 = rand()%1000+1;
			int col2 = rand()%1000+1;

			Eigen::MatrixXd h_A1 = Eigen::MatrixXd::Random(row1,col1);
			Eigen::MatrixXd h_A2 = Eigen::MatrixXd::Random(row2,col2);

			Eigen::VectorXd h_B1 = Eigen::VectorXd::Random(col1);
			Eigen::VectorXd h_B2 = Eigen::VectorXd::Random(row2);

			Matrix<double> d_A1(h_A1), d_A2(h_A2);
			Vector<double> d_B1(h_B1), d_B2(h_B2);

			Vector<double> d_C1, d_C2;
		
			d_C1 =  d_A1*d_B1;
			d_C2 =  d_A2.transpose()*d_B2;

			Eigen::VectorXd h_C1 = h_A1*h_B1;
			Eigen::VectorXd h_C2 = h_A2.transpose()*h_B2;

			ensure(check_diff(h_C1, d_C1));
			ensure(check_diff(h_C2, d_C2));

		}

		
	}

	// Test Rowwise Sum
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

			Vector<double> d_B = d_A.rowwise().sum();

			Eigen::VectorXd h_B = h_A.rowwise().sum();

			ensure(check_diff(h_B, d_B));

		}
	}

		// Test Colwise Sum
	template<>
	template<>
	void object::test<3>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;

			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row,col);

			Matrix<double> d_A(h_A);

			Vector<double> d_B = d_A.colwise().sum();

			Eigen::VectorXd h_B = h_A.colwise().sum();

			ensure(check_diff(h_B, d_B));

		}
	}

	// Test colwise Add Eq
	template<>
	template<>
	void object::test<4>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;

			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row,col);
			Eigen::VectorXd h_B = Eigen::VectorXd::Random(row);

			Matrix<double> d_A(h_A);
			Vector<double> d_B(h_B);

			d_A.colwise() += d_B;
			h_A.colwise() += h_B;

			ensure(check_diff(h_A, d_A));

		}
	}

		// Test rowwise Add Eq
	template<>
	template<>
	void object::test<5>()
	{
		for (int i = 0;i<10;i++)
		{
			int row = rand()%1000+1;
			int col = rand()%1000+1;

			Eigen::MatrixXd h_A = Eigen::MatrixXd::Random(row,col);
			Eigen::VectorXd h_B = Eigen::VectorXd::Random(col);

			Matrix<double> d_A(h_A);
			Vector<double> d_B(h_B);

			d_A.rowwise() += d_B;
			h_A.rowwise() += h_B.transpose();

			ensure(check_diff(h_A, d_A));

		}
	}

	// Test Squared Dist
	template<>
	template<>
	void object::test<6>()
	{
		for (int i = 0;i<10;i++)
		{
			int dim = rand()%1000+2;
			int samplenum1 = rand()%1000+100;
			int samplenum2 = rand()%1000+100;

			Eigen::MatrixXd ha = Eigen::MatrixXd::Random(dim,samplenum1);
			Eigen::MatrixXd hb = Eigen::MatrixXd::Random(dim,samplenum2);



			Eigen::VectorXd haa = (ha.array()*ha.array()).colwise().sum();
			Eigen::VectorXd hbb = (hb.array()*hb.array()).colwise().sum();

			Eigen::MatrixXd hdist = -2*ha.transpose()*hb;

			hdist.colwise() += haa;

			hdist.rowwise() += hbb.transpose();

			
			Matrix<double> ga(ha),gb(hb);

			Matrix<double> gdist = -2*ga.transpose()*gb;

			Vector<double> gaa = (ga.array()*ga.array()).colwise().sum();
			Vector<double> gbb = (gb.array()*gb.array()).colwise().sum();

			gdist.colwise() += gaa;
			gdist.rowwise() += gbb;

			ensure(check_diff(hdist,gdist));
		}
	}

}