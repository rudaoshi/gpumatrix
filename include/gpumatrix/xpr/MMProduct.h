/*
* Tiny Vector Matrix Library
* Dense Vector Matrix Libary of Tiny size using Expression Templates
*
* Copyright (C) 2001 - 2007 Olaf Petzold <opetzold@users.sourceforge.net>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*
* $Id: MMProduct.h,v 1.24 2007-06-23 15:58:59 opetzold Exp $
*/

#ifndef TVMET_XPR_MMPRODUCT_H
#define TVMET_XPR_MMPRODUCT_H

//#include <gpumatrix/meta/Gemm.h>
//#include <gpumatrix/loop/Gemm.h>
#include <gpumatrix/xpr/BinOpBase.h>

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
	template<class E1, class E2>
	class XprMMProduct
		: public XprBinOpBase<E1,E2,XprMMProduct<E1,E2> >, public GpuMatrixBase< XprMMProduct<E1, E2> >
	{
	private:
		XprMMProduct();
		XprMMProduct& operator=(const XprMMProduct&);
		
		using XprBinOpBase<E1,E2,XprMMProduct<E1,E2> >::m_lhs;
		using XprBinOpBase<E1,E2,XprMMProduct<E1,E2> >::m_rhs;

	public:
		typedef typename PromoteTraits<typename E1::value_type,typename E2::value_type>::value_type	value_type;
		typedef typename XprResultType<XprMMProduct<E1,E2>>::result_type result_type;
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

		std::size_t rows() const 
		{
			return m_lhs.rows();
		}

		std::size_t cols() const 
		{
			return m_rhs.cols();
		}

		std::size_t size() const 
		{
			return m_lhs.rows()*m_rhs.cols();
		}



	public:
		/** Constructor. */
		explicit XprMMProduct(const E1& lhs, const E2& rhs)
			: XprBinOpBase<E1,E2,XprMMProduct<E1,E2> >(lhs,rhs)
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
		///** index operator for arrays/matrices */
		//value_type operator()(std::size_t i, std::size_t j) const {
		//  TVMET_RT_CONDITION((i < Rows1) && (j < Cols2), "XprMMProduct Bounce Violation")
		//  return do_gemm(dispatch<use_meta>(), m_lhs, m_rhs, i, j);
		//}

		result_type eval() const
		{
			return impl::eval(*this);

		}

	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l++)
				<< "XprMMProduct<"
				<< std::endl;
			m_lhs.print_xpr(os, l);
			os << IndentLevel(l)
				<< "R1=" << m_lhs.rows() << ", C1=" << m_lhs.cols() << ",\n";
			m_rhs.print_xpr(os, l);
			os << IndentLevel(l)
				<< "C2=" << m_rhs.cols() << ",\n";
			os << IndentLevel(--l)
				<< ">," << std::endl;
		}


		//private:
		//  const E1		 				m_lhs;
		//  const E2		 				m_rhs;
	};




} // namespace gpumatrix

#endif // TVMET_XPR_MMPRODUCT_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
