#ifndef XPR_SIMPLIFY_H
#define XPR_SIMPLIFY_H

namespace gpumatrix
{
	template <class T> class Matrix;
	template <class T> class Vector;
	template <class T> class MatrixConstReference;
	template <class T> class VectorConstReference;
	template <class T> class XprMatrix;
	template <class T> class XprVector;
	template <class T> class XprLiteral;
	template<typename BinOp, typename E1, typename E2> class XprBinOp;

	template<typename E1, typename E2>	class XprMMProduct;
	template<typename E1, typename E2>	class XprMtMtProduct;
	template<typename E1, typename E2>	class XprMVProduct;

	template <class E> class XprResultType;
	template <class E> class XprMatrixTranspose;

	template <typename E>
	class Simplify
	{
	public:
		typedef E result_type;

		static result_type simplify(const E & expr) 
		{
			return expr;
		}
	};

	// (M^T)^T = M
	template <typename E>
	class Simplify<XprMatrixTranspose<XprMatrix<XprMatrixTranspose<XprMatrix<E>>>>>
	{
	public:
		typedef E result_type;

		static result_type simplify(const XprMatrixTranspose<XprMatrix<XprMatrixTranspose<XprMatrix<E>>>> & expr) 
		{
			return expr.expr().expr().expr().expr();
		}
	};

	// (A*B)^T = B^T*A^T
	template <typename E1, typename E2>
	class Simplify<XprMatrixTranspose<XprMatrix<XprMMProduct<E1,E2>>>>
	{
	public:
		typedef XprMtMtProduct<E1,E2> result_type;

		static result_type simplify(const XprMatrixTranspose<XprMatrix<XprMMProduct<E1,E2>>> & expr) 
		{
			return result_type(expr.expr().expr().rhs(), expr.expr().expr().lhs());
		}
	};

	// (A^T*B^T)^T = B*A
		template <typename E1, typename E2>
	class Simplify<XprMatrixTranspose<XprMatrix<XprMtMtProduct<E1,E2>>>>
	{
	public:
		typedef XprMMProduct<E1,E2> result_type;

		static result_type simplify(const XprMatrixTranspose<XprMatrix<XprMtMtProduct<E1,E2>>> & expr) 
		{
			return result_type(expr.expr().expr().rhs(), expr.expr().expr().lhs());
		}
	};

}

#endif