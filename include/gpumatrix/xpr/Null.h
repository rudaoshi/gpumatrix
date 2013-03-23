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
 * $Id: Null.h,v 1.11 2007-06-23 15:59:00 opetzold Exp $
 */

#ifndef TVMET_XPR_NULL_H
#define TVMET_XPR_NULL_H

namespace gpumatrix {


/**
 * \class XprNull Null.h "gpumatrix/xpr/Null.h"
 * \brief Null object design pattern
 */
class XprNull
  : public GpuMatrixBase< XprNull >
{
  XprNull& operator=(const XprNull&);

public:
  explicit XprNull() { }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l) << "XprNull[O=0]" << std::endl;
  }
};


#define TVMET_BINARY_OPERATOR(OP)                   			\
template< class T >                    					\
inline                     						\
T operator OP (const T& lhs, XprNull) { return lhs; }

TVMET_BINARY_OPERATOR(+)
TVMET_BINARY_OPERATOR(-)
TVMET_BINARY_OPERATOR(*)
TVMET_BINARY_OPERATOR(/)

#undef TVMET_BINARY_OPERATOR


} // namespace gpumatrix

#endif // TVMET_XPR_NULL_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
