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

#ifndef TVMET_ARRAYMAP_H
#define TVMET_ARRAYMAP_H

#include <iterator>					// reverse_iterator

#include <gpumatrix/gpumatrix.h>
#include <gpumatrix/TypePromotion.h>
#include <gpumatrix/RunTimeError.h>
#include <gpumatrix/xpr/ResultType.h>
#include <gpumatrix/xpr/Array.h>
#include <gpumatrix/Functional.h>
#include <gpumatrix/xpr/UnOperator.h>
#include <boost/static_assert.hpp>

#include <gpumatrix/impl/Interface.h>

namespace gpumatrix {


	/* forwards */
	template<class T/**/> class Matrix;

	namespace gpu
	{
		template <typename E, typename Dest,typename Assign> 
		void do_assign(Dest& dest, const E & e, const Assign& assign_fn);

	}
	

	/**
	* \class Matrix Matrix.h "gpumatrix/Matrix.h"
	* \brief A tiny matrix class.
	*
	* The array syntax A[j][j] isn't supported here. The reason is that
	* operator[] always takes exactly one parameter, but operator() can
	* take any number of parameters (in the case of a rectangular matrix,
	* two paramters are needed). Therefore the cleanest way to do it is
	* with operator() rather than with operator[]. \see C++ FAQ Lite 13.8
	*/
	template<class T, int D>
	class Map<Array<T,D>>
	{
	public:
		/** Data type of the gpumatrix::Matrix. */
		typedef T						value_type;

		/** Reference type of the gpumatrix::Matrix data elements. */
		typedef T&     					reference;

		/** const reference type of the gpumatrix::Matrix data elements. */
		typedef const T&     					const_reference;

		/** STL iterator interface. */
		typedef T*     					iterator;

		/** STL const_iterator interface. */
		typedef const T*     					const_iterator;

		/** STL reverse iterator interface. */
		typedef std::reverse_iterator<iterator> 		reverse_iterator;

		/** STL const reverse iterator interface. */
		typedef std::reverse_iterator<const_iterator> 	const_reverse_iterator;

	public:
		/** Dimensions. */
		//enum {
		//  Rows = NRows,			/**< Number of rows. */
		//  Cols = NCols,			/**< Number of cols. */
		//  Size = Rows * Cols			/**< Complete Size of Matrix. */
		//};

		std::size_t Rows;
		std::size_t Cols;


	public:
		/** Complexity counter. */
		//enum {
		//  ops_assign = Rows * Cols,
		//  ops        = ops_assign,
		//  use_meta   = ops < TVMET_COMPLEXITY_M_ASSIGN_TRIGGER ? true : false
		//};

	public: // STL  interface
		/** STL iterator interface. */
		iterator begin() { return m_data; }

		/** STL iterator interface. */
		iterator end() { return m_data + Rows*Cols;; }

		/** STL const_iterator interface. */
		const_iterator begin() const { return m_data; }

		/** STL const_iterator interface. */
		const_iterator end() const { return m_data + Rows*Cols;; }

		/** STL reverse iterator interface reverse begin. */
		reverse_iterator rbegin() { return reverse_iterator( end() ); }

		/** STL const reverse iterator interface reverse begin. */
		const_reverse_iterator rbegin() const {
			return const_reverse_iterator( end() );
		}

		/** STL reverse iterator interface reverse end. */
		reverse_iterator rend() { return reverse_iterator( begin() ); }

		/** STL const reverse iterator interface reverse end. */
		const_reverse_iterator rend() const {
			return const_reverse_iterator( begin() );
		}

		/** The size of the matrix. */
		std::size_t size() const { return Rows*Cols;; }

		/** STL vector max_size() - returns allways rows()*cols(). */
		std::size_t max_size() const { return  Rows*Cols;; }

		/** STL vector empty() - returns allways false. */
		bool empty() const
		{ 
			if (m_data == 0)
				return true; 
			else 
				return false;
		}

	public:
		/** The number of rows of matrix. */
		std::size_t rows() const { return Rows; }

		/** The number of columns of matrix. */
		std::size_t cols() const { return Cols; }


	public:
		/** Default Destructor */
		~Map() {
		}

		/** Default Constructor. The allocated memory region isn't cleared. If you want
		a clean use the constructor argument zero. */
		explicit Map(Array<T,D> & array):m_data(array.data()),Rows(array.rows()),Cols(array.cols())
		{ 
		}

		explicit Map(value_type * data, std::size_t row, std::size_t col):m_data(data),Rows(row),Cols(col)
		{ 
		}

		explicit Map(value_type * data, std::size_t size):m_data(data),Rows(size),Cols(1)
		{ 
			BOOST_STATIC_ASSERT (D == 1) ;
		}
		/**
		* Constructor with STL iterator interface. The data will be copied into the matrix
		* self, there isn't any stored reference to the array pointer.
		*/
		//  template<class InputIterator>
		//  explicit Matrix(InputIterator first, InputIterator last)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_RT_CONDITION(static_cast<std::size_t>(std::distance(first, last)) <= Size,
		//		       "InputIterator doesn't fits in size" )
		//    std::copy(first, last, m_data);
		//  }

		/**
		* Constructor with STL iterator interface. The data will be copied into the matrix
		* self, there isn't any stored reference to the array pointer.
		*/
		//  template<class InputIterator>
		//  explicit Matrix(InputIterator first, std::size_t sz)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_RT_CONDITION(sz <= Size, "InputIterator doesn't fits in size" )
		//    std::copy(first, first + sz, m_data);
		//  }

		/** Construct the matrix by value. */
		//  explicit Matrix(value_type rhs)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    typedef XprLiteral<value_type> expr_type;
		//    *this = XprMatrix<expr_type>(expr_type(rhs));
		//  }


		Map<Matrix<T> > matrix ()
		{
			return Map<Matrix<T>>(m_data,Rows,Cols);
		}

		NoAliasProxy<Map<Matrix<T>>> noalias()
		{
			return NoAliasProxy<Map<Matrix<T>>>(*this);
		}

	public: // access operators
		value_type* _tvmet_restrict data() { return m_data; }
		const value_type* _tvmet_restrict data() const { return m_data; }

	public: // index access operators
		//value_type& _tvmet_restrict operator()(std::size_t i, std::size_t j) {
		//	// Note: g++-2.95.3 does have problems on typedef reference
		//	TVMET_RT_CONDITION((i < Rows) && (j < Cols), "Matrix Bounce Violation")
		//		return m_data[i * Cols + j];
		//}

		//value_type operator()(std::size_t i, std::size_t j) const {
		//	TVMET_RT_CONDITION((i < Rows) && (j < Cols), "Matrix Bounce Violation")
		//		return m_data[i * Cols + j];
		//}

	public: // ET interface
		typedef ArrayConstReference<T,D>   	ConstReference;

		//typedef MatrixSliceConstReference<
		//	T,
		//	0, 0,
		//	Rows, 1
		//>							SliceConstReference;

		/** Return a const Reference of the internal data */
		ConstReference const_ref() const { return ConstReference(m_data,Rows,Cols); }

		/**
		* Return a sliced const Reference of the internal data.
		* \note Doesn't work since isn't implemented, but it is in
		* progress. Therefore this is a placeholder. */
		//ConstReference const_sliceref() const { return SliceConstReference(*this); }

		/** Return the vector as const expression. */
		XprArray<ConstReference,D> as_expr() const {
			return XprArray<ConstReference,D>(this->const_ref());
		}

	private:
		///** Wrapper for meta assign. */
		//template<class Dest, class Src, class Assign>
		//static inline
		//	void do_assign(dispatch<true>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//		meta::Matrix<Rows, 0, 0>::assign(dest, src, assign_fn);
		//}

		///** Wrapper for loop assign. */
		//template<class Dest, class Src, class Assign>
		//static inline
		//	void do_assign(dispatch<false>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//		loop::Matrix<Rows>::assign(dest, src, assign_fn);
		//}

	public:
		///** assign this to a matrix  of a different type T2 using
		//    the functional assign_fn. */
		template<class Assign>
		void assign_to(Array<T,D>& dest, const Assign& assign_fn) const {
			impl::do_assign(dest, this->const_ref(), assign_fn);
		}

	public:  // assign operations
		/** assign a given matrix of a different type T2 element wise
		to this matrix. The operator=(const Matrix&) is compiler
		generated. */
		Map & operator=(const Array<T,D>& rhs) {

			rhs.assign_to(*this, Fcnl_assign<value_type, T>());
			return *this;
		}

		Map& operator=(const ArrayConstReference<T,D>& rhs) {

			rhs.assign_to(*this, Fcnl_assign<value_type, T>());
			return *this;
		}

		/** assign a given XprMatrix element wise to this matrix. */
		template <class E>
		Map & operator=(const XprArray<E,D>& rhs) {

			rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
			return *this;
		}

		Map& operator+=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Map& operator-=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		Map& operator*=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}
		Map& operator/=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_div_eq<value_type,value_type>());
			return *this;
		}


		Map& operator+=(const Array<value_type,D> & m) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Map& operator-=(const Array<value_type,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		Map& operator*=(const Array<value_type,D> & m) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_compound_assign(*this, m, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}
		Map& operator/=(const Array<value_type,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_div_eq<value_type,value_type>());
			return *this;
		}

		Map& operator+=(const Map<Array<value_type,D>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Map& operator-=(const Map<Array<value_type,D>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		Map& operator*=(const Map<Array<value_type,D>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}
		Map& operator/=(const Map<Array<value_type,D>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_div_eq<value_type,value_type>());
			return *this;
		}

		template<class E> 
		Map& operator+=(const XprArray<E,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprArray<E,D>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Map& operator-=(const XprArray<E,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprArray<E,D>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Map& operator*=(const XprArray<E,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprArray<E,D>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Map& operator/=(const XprArray<E,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprArray<E,D>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_div_eq<value_type,value_type>());
			return *this;
		}

	private:
		////template<class Obj, std::size_t LEN> friend class CommaInitializer;

		///** This is a helper for assigning a comma separated initializer
		//list. It's equal to Matrix& operator=(value_type) which does
		//replace it. */
		//Map & assign_value(value_type rhs) {
		//	typedef XprLiteral<value_type> 			expr_type;
		//	*this = XprArray<expr_type,D>(expr_type(rhs));
		//	return *this;
		//}

	public:
		XprArray<
		  XprUnOp<
			Fcnl_exp<value_type>,
			XprArray<ArrayConstReference<value_type,D>,D>
		  >,D>  exp()
		{
			typedef  XprUnOp<Fcnl_exp<value_type>,XprArray<ArrayConstReference<value_type,D>,D>> op_type;
			return 	XprArray< op_type,D>(op_type(this->as_expr()));
		}

		XprArray<
		  XprUnOp<
			Fcnl_log<value_type>,
			XprArray<ArrayConstReference<value_type,D>,D>
		  >,D>  log()
		{
			typedef  XprUnOp<Fcnl_log<value_type>,XprArray<ArrayConstReference<value_type,D>,D>> op_type;
			return 	XprArray< op_type,D>(op_type(this->as_expr()));
		}

		XprArray<
		  XprUnOp<
			Fcnl_logistic<value_type>,
			XprArray<ArrayConstReference<value_type,D>,D>
		  >,D>  logistic()
		{
			typedef XprUnOp<Fcnl_logistic<value_type>,XprArray<ArrayConstReference<value_type,D>,D>> op_type;
			return 	XprArray< op_type,D>(op_type(this->as_expr()));
		}

		void setZero()
		{
			cudaMemset(m_data, 0,size()*sizeof(value_type));
		}


	public: // math operators with scalars


	public: // io
		/** Structure for info printing as Matrix<T>. */
		struct Info : public GpuMatrixBase<Info> {
			std::ostream& print_xpr(std::ostream& os) const {
				os << "Map<Array<T=" << typeid(value_type).name() << ",D=" << D 
					<<">>";
				return os;
			}
		};

		/** Get an info object of this matrix. */
		static Info info() { return Info(); }

		/** Member function for expression level printing. */
		std::ostream& print_xpr(std::ostream& os, std::size_t l=0) const
		{
		  os << IndentLevel(l++) << "MapMatrix<"
			 << typeid(T).name() << ", " << rows() << ", " << cols() << ">,"
			 << IndentLevel(--l)
			 << std::endl;

		  return os;
		}


		///** Member function for printing internal data. */
		//std::ostream& print_on(std::ostream& os) const;

	private:
		/** The data of matrix self. */

		value_type*						m_data;

	};



} // namespace gpumatrix

#include <gpumatrix/MapArrayFunctions.h>
#include <gpumatrix/MapArrayOperators.h>



#endif // TVMET_MATRIX_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
