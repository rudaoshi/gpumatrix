/*
 * $Id: Identity.h,v 1.4 2006-11-21 18:43:09 opetzold Exp $
 */

#ifndef TVMET_XPR_IDENTITY_H
#define TVMET_XPR_IDENTITY_H


namespace gpumatrix {


/**
 * \class XprIdentity Identity.h "gpumatrix/xpr/Identity.h"
 * \brief Expression for the identity matrix.
 *
 * This expression doesn't hold any other expression, it
 * simply returns 1 or 0 depends where the row and column
 * element excess is done.
 *
 * \since release 1.6.0
 * \sa identity
 */
template<class T>
struct XprIdentity
  : public GpuMatrixBase< XprIdentity<T> >
{
  XprIdentity& operator=(const XprIdentity&);

public:
  typedef T				value_type;

public:
  /** Complexity counter. */
  //enum {
  //  ops_assign = Rows * Cols,
  //  ops        = ops_assign
  //};

public:
  /** access by index. */
  value_type operator()(std::size_t i, std::size_t j) const {
    return i==j ? 1 : 0;
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprIdentity<"
       << std::endl;
    os << IndentLevel(l)
       << typeid(T).name() << std::endl;
    os << IndentLevel(--l) << ">"
       << ((l != 0) ? "," : "") << std::endl;
  }
};


} // namespace gpumatrix


#endif // TVMET_XPR_IDENTITY_H


// Local Variables:
// mode:C++
// tab-width:8
// End:
