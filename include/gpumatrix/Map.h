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
* $Id: Matrix.h,v 1.58 2007-06-23 15:58:58 opetzold Exp $
*/

#ifndef TVMET_MAP_H
#define TVMET_MAP_H

#include <iterator>					// reverse_iterator

#include <gpumatrix/gpumatrix.h>
#include <gpumatrix/TypePromotion.h>
#include <gpumatrix/RunTimeError.h>
#include <gpumatrix/xpr/ResultType.h>
#include <gpumatrix/xpr/Matrix.h>
//#include <gpumatrix/xpr/MatrixRow.h>
//#include <gpumatrix/xpr/MatrixCol.h>
//#include <gpumatrix/xpr/MatrixDiag.h>

#include <Eigen/Core>


namespace gpumatrix {


	

	template <class ContainerType> class Map
	{
	};

	

} // namespace gpumatrix

#include <gpumatrix/MapMatrix.h>
#include <gpumatrix/MapVector.h>
#include <gpumatrix/MapVector.h>

#endif // TVMET_MATRIX_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
