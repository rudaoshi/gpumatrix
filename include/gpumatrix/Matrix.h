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

#ifndef TVMET_MATRIX_H
#define TVMET_MATRIX_H

#include <iterator>					// reverse_iterator
#include <Eigen/Core>
#include <gpumatrix/gpumatrix.h>
#include <gpumatrix/TypePromotion.h>
#include <gpumatrix/RunTimeError.h>
#include <gpumatrix/xpr/ResultType.h>
#include <gpumatrix/xpr/Matrix.h>
// #include <gpumatrix/xpr/MatrixRow.h>
// #include <gpumatrix/xpr/MatrixCol.h>
// #include <gpumatrix/xpr/MatrixDiag.h>
#include <gpumatrix/xpr/RowWiseView.h>
#include <gpumatrix/xpr/ColWiseView.h>
#include <gpumatrix/NoAliasProxy.h>
#include <gpumatrix/MapArray.h>

#include <gpumatrix/impl/Interface.h>


namespace gpumatrix {


	/* forwards */
	template<class T/**/> class Matrix;
	template<class T, int D> class Array;
	template<class E> class Map;

	
	template<class T,
		std::size_t RowsBgn, std::size_t RowsEnd,
		std::size_t ColsBgn, std::size_t ColsEnd,
		std::size_t RowStride, std::size_t ColStride /*=1*/>
	class MatrixSliceConstReference; // unused here; for me only


	/**
	* \class MatrixConstReference Matrix.h "gpumatrix/Matrix.h"
	* \brief value iterator for ET
	*/
	template<class T/**/>
	class MatrixConstReference
		: public GpuMatrixBase < MatrixConstReference<T/**/> >
	{
	public:
		typedef T						value_type;
		typedef T*						pointer;
		typedef const T*					const_pointer;

		///** Dimensions. */
		//enum {
		//  Rows = NRows,			/**< Number of rows. */
		//  Cols = NCols,			/**< Number of cols. */
		//  Size = Rows * Cols			/**< Complete Size of Matrix. */
		//};

		std::size_t Rows;
		std::size_t Cols;


	public:
		///** Complexity counter. */
		//enum {
		//  ops       = Rows * Cols
		//};

	private:
		MatrixConstReference();
		MatrixConstReference& operator=(const MatrixConstReference&);

	public:
		/** Constructor. */
		explicit MatrixConstReference (const Matrix<T>& rhs)
			: m_data(rhs.data()),Rows(rhs.rows()),Cols(rhs.cols())
		{ }
		

		/** Constructor by a given memory pointer. */
		explicit MatrixConstReference(const_pointer data,std::size_t rows, std::size_t cols)
			: m_data(data),Rows(rows),Cols(cols)
		{ }

	public: // access operators
		///** access by index. */
		//value_type operator()(std::size_t i, std::size_t j) const {
		//	TVMET_RT_CONDITION((i < Rows) && (j < Cols), "MatrixConstReference Bounce Violation")
		//		// Do not Call This When Using GPU!
		//		value_type val;
		//	cublasGetVector (1, sizeof(value_type), m_data + i + j*Rows,1, &val, 1);
		//	return val;

		//}

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

		//const MatrixConstReference<value_type> & eval() const
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

		///** assign this to a matrix  of a different type T2 using
		//    the functional assign_fn. */
		template<class Assign>
		void assign_to(Matrix<T>& dest, const Assign& assign_fn) const {
			impl::do_assign(dest, *this, assign_fn);
		}

	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l)
				<< "MatrixConstReference<"
				<< "T=" << typeid(value_type).name() << ">,"
				<< std::endl;
		}

	private:
		const_pointer _tvmet_restrict 			m_data;
	};


	template <class E> class XprMatrixTranspose;

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
	template<class T/**/>
	class Matrix
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
		/** The number of rows of matrix. */
		std::size_t rows() const { return Rows; }

		/** The number of columns of matrix. */
		std::size_t cols() const { return Cols; }

		NoAliasProxy<Matrix<T>> noalias()
		{
			return NoAliasProxy<Matrix<T>>(*this);
		}

		Map<Array<T,2>> array() const
		{
			return Map<Array<T,2>>(m_data,Rows,Cols);
		}

	public:
		/** Default Destructor */
		~Matrix() {
			impl::free(m_data);
		}

		/** Default Constructor. The allocated memory region isn't cleared. If you want
		a clean use the constructor argument zero. */
		explicit Matrix():m_data(0),Rows(0),Cols(0)
		{ 
		}

		explicit Matrix(std::size_t nRows, std::size_t nCols):Rows(nRows),Cols(nCols)
		{
			m_data = impl::alloc<value_type>(Rows*Cols);
		}

		/** Copy Constructor, not explicit! */
		Matrix(const Matrix& rhs):Rows(rhs.rows()),Cols(rhs.cols())
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
			//		*this = XprMatrix<ConstReference>(rhs.as_expr());
		}


		Matrix(const Eigen::Matrix<value_type,Eigen::Dynamic,Eigen::Dynamic> & EigenMat):Rows(EigenMat.rows()),Cols(EigenMat.cols())
		{
			m_data = impl::alloc<value_type>(Rows*Cols);
			impl::set(m_data,EigenMat.data(),Rows*Cols);

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

		/** Construct a matrix by expression. */
		template<class E>
		Matrix(const XprMatrix<E>& e):Rows(e.rows()),Cols(e.cols())
		{
			m_data = impl::alloc<value_type>(Rows*Cols);

			(*this).noalias() = e;
		}

		template<class E>
		Matrix(const XprArray<E,2>& e):Rows(e.rows()),Cols(e.cols())
		{
			m_data = impl::alloc<value_type>(Rows*Cols);

			(*this).noalias() = e;
		}


		void setZero(std::size_t rows, std::size_t cols)
		{
			resize(rows,cols);

			impl::zero(m_data,size());
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
		//CommaInitializer<Matrix, Size> operator=(value_type rhs) {
		//  return CommaInitializer<Matrix, Size>(*this, rhs);
		//}


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
		typedef MatrixConstReference<T>   	ConstReference;

		//typedef MatrixSliceConstReference<
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
		XprMatrix<ConstReference> as_expr() const {
			return XprMatrix<ConstReference>(this->const_ref());
		}

		
		Map<Vector<value_type>> col(unsigned int i ) const
		{
			return Map<Vector<value_type>>(m_data+i*Rows,Rows);
		}

		RowWiseView<XprMatrix<ConstReference>> rowwise() const
		{
			return RowWiseView<XprMatrix<ConstReference>>(this->as_expr());
		}

		ColWiseView<XprMatrix<ConstReference>> colwise() const
		{
			return ColWiseView<XprMatrix<ConstReference>>(this->as_expr());
		}

		Map<Matrix<value_type>> block(int row_start_ind, int col_start_ind, size_t row_num, size_t col_num) const
		{
			if (row_start_ind != 0 || rows() != row_num)
				throw runtime_error("Only full column blocks are supported now!");

			return Map<Matrix<value_type>>(m_data+col_start_ind*rows(),rows(),col_num);
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
		template<class Dest, class Assign>
		void assign_to(Dest & dest, const Assign& assign_fn) const {
			impl::do_assign(dest, *this, assign_fn);
		}

	public:  // assign operations
		/** assign a given matrix of a different type T2 element wise
		to this matrix. The operator=(const Matrix&) is compiler
		generated. */
		Matrix& operator=(const Matrix & rhs) {
			
			/*cudaMemcpy(m_data, rhs.m_data*Cols*sizeof(value_type), cudaMemcpyDeviceToDevice);*/
			rhs.assign_to(*this, Fcnl_assign<value_type, value_type>());
			return *this;
		}

		Matrix& operator=(const MatrixConstReference<value_type>& rhs) {
			
			/*cudaMemcpy(m_data, rhs.m_data*Cols*sizeof(value_type), cudaMemcpyDeviceToDevice);*/

			rhs.assign_to(*this, Fcnl_assign<value_type, value_type>());
			return *this;
		}

		/** assign a given XprMatrix element wise to this matrix. */
		template <class E>
		Matrix& operator=(const XprMatrix<E>& rhs) {
			
			rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
			return *this;
		}

		Matrix& operator=(const Array<value_type,2>& rhs) {
			
			/*cudaMemcpy(m_data, rhs.m_data*Cols*sizeof(value_type), cudaMemcpyDeviceToDevice);*/
			rhs.assign_to(*this, Fcnl_assign<value_type, value_type>());
			return *this;
		}

		Matrix& operator=(const ArrayConstReference<value_type,2>& rhs) {
			
			/*cudaMemcpy(m_data, rhs.m_data*Cols*sizeof(value_type), cudaMemcpyDeviceToDevice);*/

			rhs.assign_to(*this, Fcnl_assign<value_type, value_type>());
			return *this;
		}

		/** assign a given XprMatrix element wise to this matrix. */
		template <class E>
		Matrix& operator=(const XprArray<E,2>& rhs) {
			
			rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
			return *this;
		}

		Matrix& operator=(const Eigen::Matrix<value_type,Eigen::Dynamic,Eigen::Dynamic> & EigenMat) 
		{
			
			resize(EigenMat.rows(),EigenMat.cols());
			impl::set(m_data,EigenMat.data(),Rows*Cols);

			return *this;
		}

		
	private:
		//template<class Obj, std::size_t LEN> friend class CommaInitializer;

		/** This is a helper for assigning a comma separated initializer
		list. It's equal to Matrix& operator=(value_type) which does
		replace it. */
		//Matrix& assign_value(value_type rhs) {
		//	typedef XprLiteral<value_type> 			expr_type;
		//	*this = XprMatrix<expr_type>(expr_type(rhs));
		//	return *this;
		//}

	public:

	XprMatrix<
	  XprMatrixTranspose<
		XprMatrix<MatrixConstReference<value_type>>
	  >
	> transpose () const
	{
		 typedef XprMatrixTranspose<
			XprMatrix<MatrixConstReference<value_type>>
		  >							expr_type;

		  return XprMatrix<expr_type>(
			expr_type(this->as_expr()));
	}

	value_type squaredNorm() const
	{
		return 	impl::squaredNorm(*this);
	}
	value_type sum() const
	{
		//return ((Eigen::MatrixXd)*this).sum();
		return impl::sum(*this);
	}


	value_type minCoeff() const
	{
		//return ((Eigen::Matrix<value_type,Eigen::Dynamic,Eigen::Dynamic>)*this).minCoeff();
		return impl::min(*this);
	}

	value_type maxCoeff() const
	{
		//return ((Eigen::Matrix<value_type,Eigen::Dynamic,Eigen::Dynamic>)*this).maxCoeff();
		return impl::max(*this);
	}


	public:

		static Matrix<value_type> Zero(size_t rows, size_t cols)
		{
			Matrix<value_type> zero_mat(rows,cols);

			impl::zero(zero_mat.data(),zero_mat.size());

			return zero_mat;
		}


	public: // math operators with scalars
		// NOTE: this meaning is clear - element wise ops even if not in ns element_wise
		//Matrix& operator+=(value_type) TVMET_CXX_ALWAYS_INLINE;
		//Matrix& operator-=(value_type) TVMET_CXX_ALWAYS_INLINE;
		//Matrix& operator*=(value_type) TVMET_CXX_ALWAYS_INLINE;
		//Matrix& operator/=(value_type) TVMET_CXX_ALWAYS_INLINE;

		//Matrix& operator%=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Matrix& operator^=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Matrix& operator&=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Matrix& operator|=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Matrix& operator<<=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
		//Matrix& operator>>=(std::size_t) TVMET_CXX_ALWAYS_INLINE;

		Matrix& operator+=(const Matrix<value_type> & m) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Matrix& operator-=(const Matrix<value_type> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}

		Matrix& operator+=(const Map<Matrix<value_type>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Matrix& operator-=(const Map<Matrix<value_type>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		
		template<class E> 
		Matrix& operator+=(const XprMatrix<E> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprMatrix<E>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Matrix& operator-=(const XprMatrix<E> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprMatrix<E>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}

		template <typename POD> 
		Matrix& operator*=(POD alpha) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_scalar_compound_assign(*this,(value_type)alpha, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}


	public: // math operators with matrizes
		// NOTE: access using the operators in ns element_wise, since that's what is does
		//template <class T2> Matrix& M_add_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_sub_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_mul_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_div_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_mod_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_xor_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_and_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_or_eq (const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_shl_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& M_shr_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;

	public: // math operators with expressions
		//// NOTE: access using the operators in ns element_wise, since that's what is does
		//template <class E> Matrix& M_add_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_sub_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_mul_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_div_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_mod_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_xor_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_and_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_or_eq (const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_shl_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& M_shr_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;

	public: // aliased math operators with expressions
		//template <class T2> Matrix& alias_assign(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& alias_add_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& alias_sub_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& alias_mul_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class T2> Matrix& alias_div_eq(const Matrix<T2>&) TVMET_CXX_ALWAYS_INLINE;

		//template <class E> Matrix& alias_assign(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& alias_add_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& alias_sub_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& alias_mul_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;
		//template <class E> Matrix& alias_div_eq(const XprMatrix<E>&) TVMET_CXX_ALWAYS_INLINE;

	public: // io
		/** Structure for info printing as Matrix<T>. */
		struct Info : public GpuMatrixBase<Info> {
			std::ostream& print_xpr(std::ostream& os) const {
				os << "Matrix<T=" << typeid(value_type).name()
					<< ", R=" << rows() << ", C=" << cols() << ">";
				return os;
			}
		};

		/** Get an info object of this matrix. */
		static Info info() { return Info(); }

		/** Member function for expression level printing. */
		std::ostream& print_xpr(std::ostream& os, std::size_t l=0) const
		{
		  os << IndentLevel(l++) << "Matrix<"
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

#include <gpumatrix/MapMatrix.h>
//#include <gpumatrix/MatrixImpl.h>
#include <gpumatrix/MatrixFunctions.h>
#include <gpumatrix/MatrixBinaryFunctions.h>
#include <gpumatrix/MatrixUnaryFunctions.h>
#include <gpumatrix/MatrixOperators.h>
#include <gpumatrix/MatrixEval.h>
#include <gpumatrix/NoAliasProxy.h>

#include <gpumatrix/impl/Implement.h>

#endif // TVMET_MATRIX_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
