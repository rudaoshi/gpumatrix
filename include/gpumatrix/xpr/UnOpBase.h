#ifndef TVMET_XPR_UNOPERATOR_BASE_H
#define TVMET_XPR_UNOPERATOR_BASE_H

#include <gpumatrix/TypePromotion.h>
#include <gpumatrix/xpr/ResultType.h>

namespace gpumatrix {


	/**
	* \class XprBinOp BinOperator.h "gpumatrix/xpr/BinOperator.h"
	* \brief Binary operators working on two sub expressions.
	*
	* On acessing using the index operator() the binary operation will be
	* evaluated at compile time.
	*/
	template<class E, class Deprived>
	class XprUnOpBase
	{

	public:

		typedef typename XprResultType<Deprived>::result_type result_type;

	public:

		explicit XprUnOpBase(const E& expr):m_expr(expr) 
		{ }


	public:

		std::size_t rows() const ;

		std::size_t cols() const ;

		std::size_t size() const ;

		result_type eval() const;

		const E & expr() const
		{
			return m_expr;
		}


	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const ;

	protected:
		const E						m_expr;

	};


} // namespace gpumatrix

#endif // TVMET_XPR_BINOPERATOR_BASE_H