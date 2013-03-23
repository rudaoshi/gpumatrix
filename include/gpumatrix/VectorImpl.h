/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2007 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * lesser General Public License for more details.
 *
 * You should have received a copy of the GNU lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: VectorImpl.h,v 1.31 2007-06-23 15:58:58 opetzold Exp $
 */

#ifndef TVMET_VECTOR_IMPL_H
#define TVMET_VECTOR_IMPL_H

#include <iomanip>			// setw

#include <gpumatrix/Functional.h>
#include <gpumatrix/Io.h>


namespace gpumatrix {


/*
 * member operators for i/o
 */
template<class T>
std::ostream& Vector<T>::print_xpr(std::ostream& os, std::size_t l) const
{
  os << IndentLevel(l++) << "Vector[" << ops << "]<"
     << typeid(T).name() << ", " << Size << ">,"
     << IndentLevel(--l)
     << std::endl;

  return os;
}

//
//template<class T>
//std::ostream& Vector<T>::print_on(std::ostream& os) const
//{
//  enum {
//    complex_type = NumericTraits<value_type>::is_complex
//  };
//
//  std::streamsize w = IoPrintHelper<Vector>::width(dispatch<complex_type>(), *this);
//
//  os << std::setw(0) << "[\n  ";
//  for(std::size_t i = 0; i < (Size - 1); ++i) {
//    os << std::setw(w) << m_data[i] << ", ";
//  }
//  os << std::setw(w) << m_data[Size - 1] << "\n]";
//
//  return os;
//}



} // namespace gpumatrix

#endif // TVMET_VECTOR_IMPL_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
