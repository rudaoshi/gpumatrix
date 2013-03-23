/*
* Tiny Vector Array Library
* Dense Vector Array Libary of Tiny size using Expression Templates
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
* $Id: Array.h,v 1.58 2007-06-23 15:58:58 opetzold Exp $
*/

#ifndef TVMET_ARRAY_H
#define TVMET_ARRAY_H

#include <iterator>					// reverse_iterator

#include <gpumatrix/gpumatrix.h>
#include <gpumatrix/TypePromotion.h>
#include <gpumatrix/RunTimeError.h>

#include <gpumatrix/xpr/ResultType.h>
#include <gpumatrix/xpr/Array.h>

#include <gpumatrix/NoAliasProxy.h>

#include <Eigen/Core>
#include <gpumatrix/impl/Interface.h>

namespace gpumatrix {


	/* forwards */
	template<class T, int D> class Array;
	template<class T> class Matrix;
	template<class C> class Map;
	
	//template<class T,
	//	std::size_t RowsBgn, std::size_t RowsEnd,
	//	std::size_t ColsBgn, std::size_t ColsEnd,
	//	std::size_t RowStride, std::size_t ColStride /*=1*/>
	//class ArraySliceConstReference; // unused here; for me only

	/**
	* \class ArrayConstReference Array.h "gpumatrix/Array.h"
	* \brief value iterator for ET
	*/
	template<class T, int D>
	class ArrayConstReference
		: public GpuMatrixBase < ArrayConstReference<T,D> >
	{
	public:
		typedef T						value_type;
		typedef T*						pointer;
		typedef const T*					const_pointer;

		///** Dimensions. */
		//enum {
		//  Rows = NRows,			/**< Number of rows. */
		//  Cols = NCols,			/**< Number of cols. */
		//  Size = Rows * Cols			/**< Complete Size of Array. */
		//};

		std::size_t Rows;
		std::size_t Cols;




	public:
		///** Complexity counter. */
		//enum {
		//  ops       = Rows * Cols
		//};

	private:
		ArrayConstReference();
		ArrayConstReference& operator=(const ArrayConstReference&);

	public:
		/** Constructor. */
		explicit ArrayConstReference(const Array<T,D>& rhs)
			: m_data(rhs.data()),Rows(rhs.rows()),Cols(rhs.cols())
		{ }

		/** Constructor by a given memory pointer. */
		explicit ArrayConstReference(const_pointer data,std::size_t rows, std::size_t cols)
			: m_data(data),Rows(rows),Cols(cols)
		{ }

	public: // access operators
		/** access by index. */
		value_type operator()(std::size_t i, std::size_t j) const {
			TVMET_RT_CONDITION((i < Rows) && (j < Cols), "ArrayConstReference Bounce Violation")
				// Do not Call This When Using GPU!
				value_type val;
			cublasGetVector (1, sizeof(value_type), m_data + i + j*Rows,1, &val, 1);
			return val;

		}

		std::size_t rows() const 
		{
			return Rows;
		}

		std::size_t cols() const 
		{
			return Cols;
		}

		std::size_t size() const
		{
			return Rows*Cols;
		}

		//const ArrayConstReference<value_type> & eval() const
		//{
		//	return *this;
		//}

		const_pointer data() const
		{
			return m_data;
		}

		value_type squaredNorm() const
		{
			return 	impl::squaredNorm(*this);
		}

		///** assign this to a Array  of a different type T2 using
		//    the functional assign_fn. */
		template<class Assign>
		void assign_to(Array<T,D>& dest, const Assign& assign_fn) const {
			impl::do_assign(dest, *this, assign_fn);
		}

	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l)
				<< "ArrayConstReference<"
				<< "T=" << typeid(value_type).name() << ">,"
				<< std::endl;
		}

	private:
		const_pointer _tvmet_restrict 			m_data;
	};



	/**
	* \class Array Array.h "gpumatrix/Array.h"
	* \brief A tiny Array class.
	*
	* The array syntax A[j][j] isn't supported here. The reason is that
	* operator[] always takes exactly one parameter, but operator() can
	* take any number of parameters (in the case of a rectangular Array,
	* two paramters are needed). Therefore the cleanest way to do it is
	* with operator() rather than with operator[]. \see C++ FAQ Lite 13.8
	*/
	template<class T, int D>
	class Array
	{
	public:
		/** Data type of the gpumatrix::Array. */
		typedef T						value_type;

		/** Reference type of the gpumatrix::Array data elements. */
		typedef T&     					reference;

		/** const reference type of the gpumatrix::Array data elements. */
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
		//  Size = Rows * Cols			/**< Complete Size of Array. */
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

		/** The size of the Array. */
		std::size_t size() const  { return Rows*Cols;; }

		/** STL vector max_size() - returns allways rows()*cols(). */
		std::size_t max_size() { return  Rows*Cols;; }

		/** STL vector empty() - returns allways false. */
		bool empty() 
		{ 
			if (m_data == 0)
				return true; 
			else 
				return false;
		}

	public:
		/** The number of rows of Array. */
		std::size_t rows() const { return Rows; }

		/** The number of columns of Array. */
		std::size_t cols() const { return Cols; }

		NoAliasProxy<Array<T,D>> noalias()
		{
			return NoAliasProxy<Array<T,D>>(*this);
		}
	public:
		/** Default Destructor */
		~Array() {
			impl::free(m_data);
		}

		/** Default Constructor. The allocated memory region isn't cleared. If you want
		a clean use the constructor argument zero. */
		explicit Array():m_data(0),Rows(0),Cols(0)
		{ 
		}

		explicit Array(std::size_t size):Rows(size),Cols(1)
		{
			m_data = impl::alloc<value_type>(Rows*Cols);

		}

		explicit Array(std::size_t nRows, std::size_t nCols):Rows(nRows),Cols(nCols)
		{

			m_data = impl::alloc<value_type>(Rows*Cols);

		}

		/** Copy Constructor, not explicit! */
		Array(const Array& rhs):Rows(rhs.rows()),Cols(rhs.cols())
		{
			if (this != &rhs)
			{
				if (rhs.data() == 0)
				{
					m_data = 0;
					return;
				}

				m_data = impl::alloc<value_type>(Rows*Cols);
				impl::copy(m_data,rhs.data(),rhs.size());
			}
			//		*this = XprArray<ConstReference>(rhs.as_expr());
		}


		Array(const Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic> & EigenArray):Rows(EigenArray.rows()),Cols(EigenArray.cols())
		{
			m_data = impl::alloc<value_type>(Rows*Cols);
			impl::set(m_data,EigenArray.data(),Rows*Cols);

		}


		/**
		* Constructor with STL iterator interface. The data will be copied into the Array
		* self, there isn't any stored reference to the array pointer.
		*/
		//  template<class InputIterator>
		//  explicit Array(InputIterator first, InputIterator last)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_RT_CONDITION(static_cast<std::size_t>(std::distance(first, last)) <= Size,
		//		       "InputIterator doesn't fits in size" )
		//    std::copy(first, last, m_data);
		//  }

		/**
		* Constructor with STL iterator interface. The data will be copied into the Array
		* self, there isn't any stored reference to the array pointer.
		*/
		//  template<class InputIterator>
		//  explicit Array(InputIterator first, std::size_t sz)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_RT_CONDITION(sz <= Size, "InputIterator doesn't fits in size" )
		//    std::copy(first, first + sz, m_data);
		//  }

		/** Construct the Array by value. */
		//  explicit Array(value_type rhs)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    typedef XprLiteral<value_type> expr_type;
		//    *this = XprArray<expr_type>(expr_type(rhs));
		//  }

		/** Construct a Array by expression. */
		template<class E>
		Array(const XprArray<E,D>& e):Rows(e.rows()),Cols(e.cols())
		{
			m_data = impl::alloc<value_type>(Rows*Cols);

			(*this).noalias() = e;
		}

		//template<class E>
		//Array(const XprMatrix<E>& e):Rows(e.rows()),Cols(e.cols())
		//{
		//	cublasStatus status = cublasAlloc (Rows*Cols, sizeof(value_type),(void **) &m_data);
		//	if (status != CUBLAS_STATUS_SUCCESS)
		//		throw std::runtime_error("GPU Memory Allocation Failed");

		//	(*this).alias() = e;
		//}

		Map<Matrix<value_type> > matrix ()
		{
			return Map<Matrix<value_type>>(m_data,Rows,Cols);
		}

		operator Eigen::Matrix<value_type,Eigen::Dynamic,Eigen::Dynamic> () const
		{

			Eigen::Matrix<value_type,Eigen::Dynamic,Eigen::Dynamic> M(Rows,Cols);

			impl::get(M.data(),m_data,size());

			return M;
		}


		void resize(std::size_t r, std::size_t c)
		{
			if(Rows == r && Cols == c)

				return;

			impl::free(m_data);

			Rows = r; Cols = c;

			m_data = impl::alloc<value_type>(Rows*Cols);

		}
		///** assign a value_type on array, this can be used for a single value
		//    or a comma separeted list of values. */
		//CommaInitializer<Array, Size> operator=(value_type rhs) {
		//  return CommaInitializer<Array, Size>(*this, rhs);
		//}

	public: // access operators
		value_type* _tvmet_restrict data() { return m_data; }
		const value_type* _tvmet_restrict data() const { return m_data; }

	public: // index access operators
		//value_type& _tvmet_restrict operator()(std::size_t i, std::size_t j) {
		//	// Note: g++-2.95.3 does have problems on typedef reference
		//	TVMET_RT_CONDITION((i < Rows) && (j < Cols), "Array Bounce Violation")
		//		return m_data[i * Cols + j];
		//}

		//value_type operator()(std::size_t i, std::size_t j) const {
		//	TVMET_RT_CONDITION((i < Rows) && (j < Cols), "Array Bounce Violation")
		//		return m_data[i * Cols + j];
		//}

	public: // ET interface
		typedef ArrayConstReference<T,D>   	ConstReference;

		//typedef ArraySliceConstReference<
		//	T,
		//	0, 0,
		//	Rows, 1
		//>							SliceConstReference;

		/** Return a const Reference of the internal data */
		ConstReference const_ref() const { return ConstReference(*this); }

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
		//		meta::Array<Rows, 0, 0>::assign(dest, src, assign_fn);
		//}

		///** Wrapper for loop assign. */
		//template<class Dest, class Src, class Assign>
		//static inline
		//	void do_assign(dispatch<false>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//		loop::Array<Rows>::assign(dest, src, assign_fn);
		//}

	public:
		///** assign this to a Array  of a different type T2 using
		//    the functional assign_fn. */
		template<class Dest, class Assign>
		void assign_to(Dest & dest, const Assign& assign_fn) const {
			impl::do_assign(dest, *this, assign_fn);
		}

	public:  // assign operations
		/** assign a given Array of a different type T2 element wise
		to this Array. The operator=(const Array&) is compiler
		generated. */
		Array& operator=(const Array & rhs) {
			resize(rhs.rows(),rhs.cols());
			/*cudaMemcpy(m_data, rhs.m_data*Cols*sizeof(value_type), cudaMemcpyDeviceToDevice);*/
			rhs.assign_to(*this, Fcnl_assign<value_type, value_type>());
			return *this;
		}

		Array& operator=(const ArrayConstReference<value_type,D>& rhs) {
			resize(rhs.rows(),rhs.cols());
			/*cudaMemcpy(m_data, rhs.m_data*Cols*sizeof(value_type), cudaMemcpyDeviceToDevice);*/

			rhs.assign_to(*this, Fcnl_assign<value_type, value_type>());
			return *this;
		}

		/** assign a given XprArray element wise to this Array. */
		template <class E>
		Array& operator=(const XprArray<E,D>& rhs) {
			resize(rhs.rows(),rhs.cols());
			rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
			return *this;
		}

	private:


		/** This is a helper for assigning a comma separated initializer
		list. It's equal to Array& operator=(value_type) which does
		replace it. */
		Array & assign_value(value_type rhs) {
			typedef XprLiteral<value_type> 			expr_type;
			*this = XprArray<expr_type,D>(expr_type(rhs));
			return *this;
		}

	public:


	public: // math operators with scalars
		// NOTE: this meaning is clear - element wise ops even if not in ns element_wise
		Array& operator+=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Array& operator-=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		Array& operator*=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}
		Array& operator/=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_div_eq<value_type,value_type>());
			return *this;
		}


		Array& operator+=(const Array<value_type,D> & m) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Array& operator-=(const Array<value_type,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		Array& operator*=(const Array<value_type,D> & m) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_compound_assign(*this, m, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}
		Array& operator/=(const Array<value_type,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_div_eq<value_type,value_type>());
			return *this;
		}

		Array& operator+=(const Map<Array<value_type,D>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Array& operator-=(const Map<Array<value_type,D>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		Array& operator*=(const Map<Array<value_type,D>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}
		Array& operator/=(const Map<Array<value_type,D>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_div_eq<value_type,value_type>());
			return *this;
		}

		template<class E> 
		Array& operator+=(const XprArray<E,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprArray<E,D>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Array& operator-=(const XprArray<E,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprArray<E,D>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Array& operator*=(const XprArray<E,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprArray<E,D>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Array& operator/=(const XprArray<E,D> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprArray<E,D>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_div_eq<value_type,value_type>());
			return *this;
		}

		XprArray<
		  XprUnOp<
			Fcnl_exp<value_type>,
			XprArray<ArrayConstReference<value_type,D>,D>
		  >,D> exp()
		{
			typedef XprUnOp<Fcnl_exp<value_type>,XprArray<ArrayConstReference<value_type,D>,D> > op_type;
			return 	XprArray< op_type,D>(op_type(this->as_expr()));
		}

		XprArray<
		  XprUnOp<
			Fcnl_logistic<value_type>,
			XprArray<ArrayConstReference<value_type,D>,D>
		  >,D> logistic()
		{
			typedef XprUnOp<Fcnl_logistic<value_type>,XprArray<ArrayConstReference<value_type,D>,D> > op_type;
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
			Fcnl_arrayinv<value_type>,
			XprArray<ArrayConstReference<value_type,D>,D>
		  >,D> inverse()
		{
			typedef XprUnOp<Fcnl_arrayinv<value_type>,XprArray<ArrayConstReference<value_type,D>,D> > op_type;
			return 	XprArray< op_type,D>(op_type(this->as_expr()));
		}

		value_type squaredNorm() const
		{
			return 	impl::squaredNorm(*this);
		}

		
		value_type sum() const
		{
			//Eigen::Matrix<value_type,Eigen::Dynamic,Eigen::Dynamic> eigarray = (Eigen::Matrix<value_type,Eigen::Dynamic,Eigen::Dynamic>)(*this);
			//return 	eigarray.sum();
			return impl::sum(*this);
		}


		void setZero(std::size_t rows, std::size_t cols)
		{
			resize(rows,cols);

			impl::zero(m_data,size());
		}

		//Array& operator%=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Array& operator^=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Array& operator&=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Array& operator|=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Array& operator<<=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Array& operator>>=(std::size_t) TVMET_CXX_ALWAYS_INLINE;

	public: // math operators with matrizes
		// NOTE: access using the operators in ns element_wise, since that's what is does
		//template <class T2> Array& M_add_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_sub_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_mul_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_div_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_mod_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_xor_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_and_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_or_eq (const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_shl_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& M_shr_eq(const Array<T2>&) TVMET_CXX_ALWAYS_INLINE;

	public: // math operators with expressions
		// NOTE: access using the operators in ns element_wise, since that's what is does
		//template <class E> Array& M_add_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_sub_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_mul_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_div_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_mod_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_xor_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_and_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_or_eq (const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_shl_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& M_shr_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;

	public: // aliased math operators with expressions
		//template <class T2> Array& alias_assign(const Array<T2,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& alias_add_eq(const Array<T2,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& alias_sub_eq(const Array<T2,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& alias_mul_eq(const Array<T2,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Array& alias_div_eq(const Array<T2,D>&) TVMET_CXX_ALWAYS_INLINE;

		//template <class E> Array& alias_assign(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& alias_add_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& alias_sub_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& alias_mul_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Array& alias_div_eq(const XprArray<E,D>&) TVMET_CXX_ALWAYS_INLINE;

	public: // io
		/** Structure for info printing as Array<T>. */
		struct Info : public GpuMatrixBase<Info> {
			std::ostream& print_xpr(std::ostream& os) const {
				os << "Array<T=" << typeid(value_type).name()
					<< ", R=" << Rows << ", C=" << Cols << ">";
				return os;
			}
		};

		/** Get an info object of this Array. */
		static Info info() { return Info(); }

		/** Member function for expression level printing. */
		std::ostream& print_xpr(std::ostream& os, std::size_t l=0) const
		{
			
		  os << IndentLevel(l++) << "Array<"
			 << typeid(T).name() << ", " << rows() << ", " << cols() << ">,"
			 << IndentLevel(--l)
			 << std::endl;

		  return os;


		}

		///** Member function for printing internal data. */
		//std::ostream& print_on(std::ostream& os) const;

	private:
		/** The data of Array self. */

		value_type*						m_data;

	};



} // namespace gpumatrix

#include <gpumatrix/MapArray.h>
#include <gpumatrix/ArrayFunctions.h>
#include <gpumatrix/ArrayOperators.h>
#include <gpumatrix/NoAliasProxy.h>


#include <gpumatrix/impl/Implement.h>
#endif // TVMET_Array_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
