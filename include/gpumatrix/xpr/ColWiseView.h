#ifndef COLWISE_VIEW_H_
#define COLWISE_VIEW_H_

#include <gpumatrix/xpr/ColWiseSum.h>
#include <gpumatrix/impl/Interface.h>
namespace gpumatrix {

	template <class T/**/> class Matrix;
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
	class ColWiseView
		: public GpuMatrixBase< ColWiseView<E> >
	{
	private:
		ColWiseView();
		ColWiseView& operator=(const ColWiseView&);

	public:
		typedef typename E::value_type	value_type;

	public:
		/** Complexity counter. */
		// enum {
		//ops_lhs   = E1::ops,
		//ops_rhs   = E2::ops,
		//M         = Rows1 * Cols1 * Cols2,
		//N         = Rows1 * (Cols1 - 1) * Cols2,
		//ops_plus  = M * NumericTraits<value_type>::ops_plus,
		//ops_muls  = N * NumericTraits<value_type>::ops_muls,
		//ops       = ops_plus + ops_muls,
		//use_meta  = Rows1*Cols2 < TVMET_COMPLEXITY_MM_TRIGGER ? true : false
		// };

		//XprVector<ColWiseView<E>> sum()
		//{
		//	return XprVector<ColWiseSum<E>>(ColWiseSum<E>(m_expr));
		//}

		typename E::result_type operator += (const Vector<value_type> & x)
		{
			typename E::result_type result = m_expr.eval();
			impl::colwise_array_compound_op(const_cast<value_type *>(result.data()) , m_expr.rows(),m_expr.cols(), x.data(), Fcnl_colwise_add_eq<value_type,value_type>());
			return result;
		}

		typename E::result_type operator += (const Map<Vector<value_type>> & x)
		{
			typename E::result_type result = m_expr.eval();
			impl::colwise_array_compound_op(const_cast<value_type *>(result.data()) , m_expr.rows(),m_expr.cols(), x.data(), Fcnl_colwise_add_eq<value_type,value_type>());
			return result;
		}


		XprVector<ColWiseSum<E>> sum()
		{
			return XprVector<ColWiseSum<E>>(ColWiseSum<E>(m_expr));
		}
		
	public:
		/** Constructor. */
		explicit ColWiseView(const E& expr)
			: m_expr(expr)
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
				<< "ColWiseView<"
				<< std::endl;
			m_expr.print_xpr(os, l);
			
			os << IndentLevel(--l)
				<< ">," << std::endl;
		}


		private:
		  const E		 				m_expr;
		//  const E2		 				m_rhs;
	};




} // namespace gpumatrix

#endif // TVMET_XPR_MMPRODUCT_H


