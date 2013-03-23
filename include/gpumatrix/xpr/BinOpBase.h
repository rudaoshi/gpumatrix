#ifndef TVMET_XPR_BINOPERATOR_BASE_H
#define TVMET_XPR_BINOPERATOR_BASE_H

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
template<class E1, class E2, class Deprived>
class XprBinOpBase
{

public:

	typedef typename XprResultType<Deprived>::result_type result_type;

public:

	explicit XprBinOpBase(const E1& lhs, const E2& rhs):m_lhs(lhs),m_rhs(rhs)
	  { }


public:
 
  std::size_t rows() const ;

  std::size_t cols() const ;

  std::size_t size() const ;

  result_type eval() const;

  const E1 & lhs() const
  {
	  return m_lhs;
  }

  const E2 & rhs() const
  {
	  return m_rhs;
  }


public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const ;

protected:
  const E1						m_lhs;
  const E2						m_rhs;
};


} // namespace gpumatrix

#endif // TVMET_XPR_BINOPERATOR_BASE_H