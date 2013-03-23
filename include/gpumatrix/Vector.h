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
* $Id: Vector.h,v 1.48 2007-06-23 15:58:58 opetzold Exp $
*/

#ifndef TVMET_VECTOR_H
#define TVMET_VECTOR_H

#include <iterator>					// reverse_iterator

#include <gpumatrix/gpumatrix.h>
#include <gpumatrix/TypePromotion.h>
//#include <gpumatrix/CommaInitializer.h>
#include <gpumatrix/RunTimeError.h>
#include <gpumatrix/xpr/ResultType.h>
#include <gpumatrix/xpr/Vector.h>

#include <gpumatrix/NoAliasProxy.h>

#include <gpumatrix/impl/Interface.h>

namespace gpumatrix {


	/* forwards */
	template<class T> class Vector;
	template<class T, int D> class Array;
	template<class E> class Map;

	/**
	* \class VectorConstReference Vector.h "gpumatrix/Vector.h"
	* \brief Const value iterator for ET
	*/
	template<class T>
	class VectorConstReference
		: public GpuMatrixBase< VectorConstReference<T> >
	{
	public: // types
		typedef T 						value_type;
		typedef T*						pointer;
		typedef const T*					const_pointer;

	public:
		//  /** Dimensions. */
		//  enum {
		//    Size = Sz			/**< The size of the vector. */
		//  };
		std::size_t Size;

	public:
		//  /** Complexity counter. */
		//  enum {
		//    ops        = Size
		//  };

	private:
		VectorConstReference();
		VectorConstReference& operator=(const VectorConstReference&);

	public:
		/** Constructor. */
		explicit VectorConstReference(const Vector<T>& rhs)
			: m_data(rhs.data()),Size(rhs.size())
		{ }

		/** Constructor by a given memory pointer. */
		explicit VectorConstReference(const_pointer data, std::size_t size)
			: m_data(data),Size(size)
		{ }

	public: // access operators
		/** access by index. */
		value_type operator()(std::size_t i) const {
			TVMET_RT_CONDITION(i < Size, "VectorConstReference Bounce Violation")
				value_type val;
			cublasGetVector (1, sizeof(value_type), m_data + i,1, &val, 1);
			return val;
		}

		template<class Assign>
		void assign_to(Vector<T>& dest, const Assign& assign_fn) const {
			impl::do_assign(dest, *this, assign_fn);
		}


		std::size_t rows() const 
		{
			return Size;
		}

		std::size_t cols() const 
		{
			return 1;
		}

		std::size_t size() const
		{
			return Size;
		}

		const_pointer data() const
		{
			return m_data;
		}

		value_type squaredNorm() const
		{
			return 	impl::squaredNorm(*this);
		}
	public: // debugging Xpr parse tree
		void print_xpr(std::ostream& os, std::size_t l=0) const {
			os << IndentLevel(l)
				<< "VectorConstReference<"
				<< "T=" << typeid(T).name() << ">,"
				<< std::endl;
		}

	private:
		const_pointer _tvmet_restrict 			m_data;
	};


	/**
	* \class Vector Vector.h "gpumatrix/Vector.h"
	* \brief Compile time fixed length vector with evaluation on compile time.
	*/
	template<class T>
	class Vector
	{
	public:
		/** Data type of the gpumatrix::Vector. */
		typedef T     					value_type;

		/** Reference type of the gpumatrix::Vector data elements. */
		typedef T&     					reference;

		/** const reference type of the gpumatrix::Vector data elements. */
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
		///** Dimensions. */
		//enum {
		//  Size = Sz			/**< The size of the vector. */
		//};

	private:
		std::size_t Size;

	public:
		/** Complexity counter. */
		//enum {
		//  ops_assign = Size,
		//  ops        = ops_assign,
		//  use_meta   = ops < TVMET_COMPLEXITY_V_ASSIGN_TRIGGER ? true : false
		//};

	public: // STL  interface
		/** STL iterator interface. */
		iterator begin() { return m_data; }

		/** STL iterator interface. */
		iterator end() { return m_data + Size; }

		/** STL const_iterator interface. */
		const_iterator begin() const { return m_data; }

		/** STL const_iterator interface. */
		const_iterator end() const { return m_data + Size; }

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

		/** STL vector front element. */
		value_type front() { return m_data[0]; }

		/** STL vector const front element. */
		const_reference front() const { return m_data[0]; }

		/** STL vector back element. */
		value_type back() { return m_data[Size-1]; }

		/** STL vector const back element. */
		const_reference back() const { return m_data[Size-1]; }

		/** STL vector empty() - returns allways false. */
		bool empty() { if (m_data == 0) return true; else return false; }

		/** The size of the vector. */
		std::size_t size() const { return Size; }

		/** STL vector max_size() - returns allways Size. */
		std::size_t max_size() { return Size; }

	public:
		/** Default Destructor */
		~Vector() {
			impl::free(m_data);
		}

		/** Default Constructor. The allocated memory region isn't cleared. If you want
		a clean use the constructor argument zero. */
		explicit Vector():m_data(0),Size(0)
		{
		}

				/** Default Constructor. The allocated memory region isn't cleared. If you want
		a clean use the constructor argument zero. */
		explicit Vector(std::size_t size):m_data(0),Size(0)
		{
			resize(size);
		}

		/** Copy Constructor, not explicit! */
		Vector(const Vector& rhs):m_data(0),Size(0)
		{
			if (this != &rhs)
			{
				if (rhs.data() == 0)
				{
					m_data = 0;
					return;
				}

				resize(rhs.size());
				impl::copy(m_data,rhs.data(),size());
//				cudaMemcpy(m_data, rhs.m_data, Size*sizeof(value_type), cudaMemcpyDeviceToDevice);
			}
			//*this = XprVector<ConstReference>(rhs.as_expr());
		}

		Vector(const Eigen::Matrix<value_type,Eigen::Dynamic,1> & EigenVec):m_data(0),Size(0)
		{
			resize(EigenVec.size());
			impl::set(m_data,EigenVec.data(),size());
//			cublasSetVector(Size,sizeof(value_type),EigenVec.data(),1,m_data,1);
		}


		/**
		* Constructor with STL iterator interface. The data will be copied into the
		* vector self, there isn't any stored reference to the array pointer.
		*/
		//  template<class InputIterator>
		//  explicit Vector(InputIterator first, InputIterator last)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_RT_CONDITION( static_cast<std::size_t>(std::distance(first, last)) <= Size,
		//			"InputIterator doesn't fits in size" )
		//    std::copy(first, last, m_data);
		//  }

		//  /**
		//   * Constructor with STL iterator interface. The data will be copied into the
		//   * vector self, there isn't any stored reference to the array pointer.
		//   */
		//  template<class InputIterator>
		//  explicit Vector(InputIterator first)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_RT_CONDITION( sz <= Size, "InputIterator doesn't fits in size" )
		//    std::copy(first, first + sz, m_data);
		//  }
		//
		//  /** Constructor with initializer for all elements.  */
		//  explicit Vector(value_type rhs)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    typedef XprLiteral<value_type> expr_type;
		//    *this = XprVector<expr_type>(expr_type(rhs));
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(2 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1;
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1, value_type x2)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(3 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2;
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(4 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3;
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		//		  value_type x4)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(5 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		//		  value_type x4, value_type x5)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(6 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
		//    m_data[5] = x5;
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		//		  value_type x4, value_type x5, value_type x6)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(7 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
		//    m_data[5] = x5; m_data[6] = x6;
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		//		  value_type x4, value_type x5, value_type x6, value_type x7)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(8 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
		//    m_data[5] = x5; m_data[6] = x6; m_data[7] = x7;
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		//		  value_type x4, value_type x5, value_type x6, value_type x7,
		//		  value_type x8)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(9 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
		//    m_data[5] = x5; m_data[6] = x6; m_data[7] = x7; m_data[8] = x8;
		//  }
		//
		//  /** Default Constructor with initializer list. */
		//  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		//		  value_type x4, value_type x5, value_type x6, value_type x7,
		//		  value_type x8, value_type x9)
		//#if defined(TVMET_DYNAMIC_MEMORY)
		//    : m_data( new value_type[Size] )
		//#endif
		//  {
		//    TVMET_CT_CONDITION(10 <= Size, ArgumentList_is_too_long)
		//    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
		//    m_data[5] = x5; m_data[6] = x6; m_data[7] = x7; m_data[8] = x8; m_data[9] = x9;
		//  }

		/** Construct a vector by expression. */
		template <class E>
		Vector(const XprVector<E>& e):m_data(0),Size(0)
		{
			resize(e.size());

			noalias() = e;
		}


		std::size_t rows() const 
		{
			return Size;
		}

		std::size_t cols() const 
		{
			return 1;
		}
		void resize(std::size_t size)
		{
			if(Size == size)

				return;

			impl::free(m_data);

			Size = size;

			m_data = impl::alloc<value_type>(Size);


		}

		///** Assign a value_type on array, this can be used for a single value
		//    or a comma separeted list of values. */
		//CommaInitializer<Vector> operator=(value_type rhs) {
		//  return CommaInitializer<Vector>(*this, rhs);
		//}

		Vector& operator*=(value_type alpha) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_scalar_compound_assign(*this, alpha, Fcnl_mul_eq<value_type,value_type>());
			return *this;
		}

		Vector& operator+=(const Vector<value_type> & m) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Vector& operator-=(const Vector<value_type> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}

		Vector& operator+=(const Map<Vector<value_type>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Vector& operator-=(const Map<Vector<value_type>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		
		template<class E> 
		Vector& operator+=(const XprVector<E> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprMatrix<E>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Vector& operator-=(const XprVector<E> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprMatrix<E>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}

	public: // access operators
		value_type* _tvmet_restrict data() { return m_data; }
		const value_type* _tvmet_restrict data() const { return m_data; }

	public: // index access operators
		//value_type& _tvmet_restrict operator()(std::size_t i) {
		//	// Note: g++-2.95.3 does have problems on typedef reference
		//	TVMET_RT_CONDITION(i < Size, "Vector Bounce Violation")
		//		return m_data[i];
		//}

		//value_type operator()(std::size_t i) const {
		//	TVMET_RT_CONDITION(i < Size, "Vector Bounce Violation")
		//		return m_data[i];
		//}

		//value_type& _tvmet_restrict operator[](std::size_t i) {
		//	// Note: g++-2.95.3 does have problems on typedef reference
		//	return this->operator()(i);
		//}

		//value_type operator[](std::size_t i) const {
		//	return this->operator()(i);
		//}

	public: // ET interface
		typedef VectorConstReference<T>    		ConstReference;

		/** Return a const Reference of the internal data */
		ConstReference const_ref() const { return ConstReference(*this); }

		/** Return the vector as const expression. */
		XprVector<ConstReference> as_expr() const {
			return XprVector<ConstReference>(this->const_ref());
		}

		
		NoAliasProxy<Vector<T>> noalias()
		{
			return NoAliasProxy<Vector<T>>(*this);
		}

		Map<Array<T,1>> array()
		{
			return Map<Array<T,1>>(m_data,Size);
		}


	private:
		///** Wrapper for meta assign. */
		//template<class Dest, class Src, class Assign>
		//static inline
		//void do_assign(dispatch<true>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//  meta::Vector<Size, 0>::assign(dest, src, assign_fn);
		//}

		///** Wrapper for loop assign. */
		//template<class Dest, class Src, class Assign>
		//static inline
		//void do_assign(dispatch<false>, Dest& dest, const Src& src, const Assign& assign_fn) {
		//  loop::Vector<Size>::assign(dest, src, assign_fn);
		//}

	public:
		/** assign this to a vector expression using the functional assign_fn. */
		template<class Dest, class Assign>
		void assign_to(Dest & dest, const Assign& assign_fn) const {
			impl::do_assign(dest, *this, assign_fn);
		}

	public:   // assign operations
		/** assign a given Vector element wise to this vector.
		The operator=(const Vector&) is compiler generated. */
		Vector& operator = (const Vector & rhs) {
			//resize(rhs.size());
			rhs.assign_to(*this, Fcnl_assign<value_type, value_type>());
			return *this;
		}

		/** assign a given XprVector element wise to this vector. */
		template<class E>
		Vector& operator=(const XprVector<E>& rhs) {
			//resize(rhs.size());
			rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
			return *this;
		}


		Vector& operator=(const VectorConstReference<value_type>& rhs) {
			//resize(rhs.size());

			rhs.assign_to(*this, Fcnl_assign<value_type, value_type>());
			return *this;
		}

		operator Eigen::Matrix<value_type,Eigen::Dynamic,1> () const
		{

			Eigen::Matrix<value_type,Eigen::Dynamic,1> M(Size);

			impl::get(M.data(),m_data,Size);
			return M;
		}

		void setZero(std::size_t s)
		{
			resize(s);

			impl::zero(m_data,size());
		}

		value_type squaredNorm() const
		{
			return 	impl::squaredNorm(*this);
		}

		value_type sum() const
		{
			return 	impl::sum(*this);
		}

		value_type dot(const Vector<value_type> & other) const
		{
			return impl::dot(*this, other);
		}


	private:
		//template<class Obj, std::size_t LEN> friend class CommaInitializer;

		///** This is a helper for assigning a comma separated initializer
		//list. It's equal to Vector& operator=(value_type) which does
		//replace it. */
		//Vector& assign_value(value_type rhs) {
		//	typedef XprLiteral<value_type> 			expr_type;
		//	*this = XprVector<expr_type>(expr_type(rhs));
		//	return *this;
		//}

	public: // math operators with scalars

	public: // io
		/** Structure for info printing as Vector<T>. */
		struct Info : public GpuMatrixBase<Info> {
			std::ostream& print_xpr(std::ostream& os) const {
				os << "Vector<T=" << typeid(value_type).name()
					<< "=" << Size << ">";
				return os;
			}
		};

		/** Get an info object of this vector. */
		static Info info() { return Info(); }

		/** Member function for expression level printing. */
		std::ostream& print_xpr(std::ostream& os, std::size_t l=0) const
		{
		  os << IndentLevel(l++) << "Vector<"
			 << typeid(T).name() << ", " << size() << ">,"
			 << IndentLevel(--l)
			 << std::endl;

		  return os;
		}

		///** Member function for printing internal data. */
		//std::ostream& print_on(std::ostream& os) const;

	private:
		/** The data of vector self. */


		value_type*	m_data;

	};


} // namespace gpumatrix

#include <gpumatrix/MapVector.h>
//#include <gpumatrix/VectorImpl.h>
#include <gpumatrix/VectorFunctions.h>
#include <gpumatrix/VectorBinaryFunctions.h>
#include <gpumatrix/VectorUnaryFunctions.h>
#include <gpumatrix/VectorOperators.h>
#include <gpumatrix/NoAliasProxy.h>

#include <gpumatrix/impl/Implement.h>

#endif // TVMET_VECTOR_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
