#ifndef COLWISE_SUM_H_
#define COLWISE_SUM_H_

#include <gpumatrix/xpr/UnOpBase.h>


namespace gpumatrix {

	template <class T/**/> class Vector;
	/**
	* \class XprMMProduct MMProduct.h "gpumatrix/xpr/MMProduct.h"
	* \brief Expression for matrix-matrix product.
	*        Using formula:
	*        \f[
	*        M_1\,M_2
	*        \f]
	* \note The Rows2 has to be  equal to Cols1.
	*/
	template<class E>
	class ColWiseSum
		: public XprUnOpBase<E,ColWiseSum<E>>, public GpuMatrixBase< ColWiseSum<E> >
	{
	private:
		ColWiseSum();
		ColWiseSum& operator=(const ColWiseSum&);
		
		using XprUnOpBase<E,ColWiseSum<E>>::m_expr;

	public:
		typedef typename E::value_type	value_type;
		typedef Vector<value_type> result_type;

	public:

		std::size_t rows() const 
		{
			return m_expr.cols();
		}

		std::size_t cols() const 
		{
			return 1;
		}

		std::size_t size() const 
		{
			return m_expr.cols();
		}

		result_type eval() const
		{
			return impl::eval(*this);
		}

	public:
		/** Constructor. */
		explicit ColWiseSum(const E& expr)
			: XprUnOpBase<E,ColWiseSum<E>>(expr)
		{ }

		/** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
		XprMMProduct(const XprMMProduct& e)
			: m_lhs(e.m_lhs), m_rhs(e.m_rhs)
		{ }
#endif

	private:
		///** Wrapper for meta gemm. */
		//static inline
		//value_type do_gemm(dispatch<true>, const E1& lhs, const E2& rhs, std::size_t i, std::size_t j) {
		//  return meta::gemm<Rows1,
		//                    Cols2,
		//                    0>::prod(lhs, rhs, i, j);
		//}

		///** Wrapper for loop gemm. */
		//static inline
		//value_type do_gemm(dispatch<false>, const E1& lhs, const E2& rhs, std::size_t i, std::size_t j) {
		//  return loop::gemm<Rows1>::prod(lhs, rhs, i, j);
		//}

	public:


	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l++)
				<< "ColWiseSum<"
				<< std::endl;
			os << IndentLevel(l)
				<< "Row=" << rows()<< ",\n";

			os << IndentLevel(--l)
				<< ">," << std::endl;
		}



	};




} // namespace gpumatrix

#endif // TVMET_XPR_MMPRODUCT_H


