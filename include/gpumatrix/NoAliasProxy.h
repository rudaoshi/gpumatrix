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
 * $Id: AliasProxy.h,v 1.8 2007-06-23 15:58:58 opetzold Exp $
 */

#ifndef TVMET_ALIAS_PROXY_H
#define TVMET_ALIAS_PROXY_H

namespace gpumatrix {


/** forwards */
template<class E> class NoAliasProxy;


/**
 * \class AliasProxy AliasProxy.h "gpumatrix/AliasProxy.h"
 * \brief Assign proxy for alias Matrices and Vectors.
 *
 *        A short lived object to provide simplified alias syntax.
 *        Only the friend function alias is allowed to create
 *        such a object. The proxy calls the appropriate member
 *        alias_xyz() which have to use temporaries to avoid
 *        overlapping memory regions.
 * \sa alias
 * \sa Some Notes \ref alias
 * \note Thanks to ublas-dev group, where the principle idea
 *       comes from.
 */
template<class E>
class NoAliasProxy
{

  NoAliasProxy& operator=(const NoAliasProxy&);

public:
  NoAliasProxy(E& lord) : m_lord(lord) { }


	typedef typename E::value_type value_type;

	/** assign a given XprMatrix element wise to this matrix. */
	template <class E2>
	void operator=(const E2 & rhs) {
		rhs.assign_to(*this, Fcnl_assign<value_type, typename E2::value_type>());
	}

  E &  lord()
  {
	  return m_lord;
  }

private:
  E&	m_lord;
};




} // namespace gpumatrix


#endif /* TVMET_ALIAS_PROXY_H */

// Local Variables:
// mode:C++
// tab-width:8
// End:
