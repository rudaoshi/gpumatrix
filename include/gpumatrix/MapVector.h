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

#ifndef TVMET_MAP_VECTOR_H
#define TVMET_MAP_VECTOR_H

#include <iterator>					// reverse_iterator

#include <gpumatrix/gpumatrix.h>
#include <gpumatrix/TypePromotion.h>
//#include <gpumatrix/CommaInitializer.h>
#include <gpumatrix/RunTimeError.h>
#include <gpumatrix/xpr/ResultType.h>
#include <gpumatrix/xpr/Vector.h>

#include <gpumatrix/NoAliasProxy.h>

namespace gpumatrix {


	/* forwards */
	template<class T> class Vector;



	/**
	* \class Vector Vector.h "gpumatrix/Vector.h"
	* \brief Compile time fixed length vector with evaluation on compile time.
	*/
	template<class T>
	class Map<Vector<T>>
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
		~Map() {
			
		}


				/** Default Constructor. The allocated memory region isn't cleared. If you want
		a clean use the constructor argument zero. */
		explicit Map(value_type * data, std::size_t size):m_data(data),Size(size)
		{ 
		}


		/** Construct a vector by expression. */


		std::size_t rows() const 
		{
			return Size;
		}

		std::size_t cols() const 
		{
			return 1;
		}
		

		///** Assign a value_type on array, this can be used for a single value
		//    or a comma separeted list of values. */
		//CommaInitializer<Vector> operator=(value_type rhs) {
		//  return CommaInitializer<Vector>(*this, rhs);
		//}

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
		ConstReference const_ref() const { return ConstReference(m_data,Size); }

		/** Return the vector as const expression. */
		XprVector<ConstReference> as_expr() const {
			return XprVector<ConstReference>(this->const_ref());
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
		template<class T2>
		Map & operator=(const Vector<T2>& rhs) {
			//resize(rhs.size());
			rhs.assign_to(*this, Fcnl_assign<value_type, T2>());
			return *this;
		}

		/** assign a given XprVector element wise to this vector. */
		template<class E>
		Map & operator=(const XprVector<E>& rhs) {
			//resize(rhs.size());
			rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
			return *this;
		}

		/** assign a given XprVector element wise to this vector. */
		template<class E>
		Map & operator=(const XprArray<E,1>& rhs) {
			//resize(rhs.size());
			rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
			return *this;
		}

		template<class T2>
		Map & operator=(const VectorConstReference<T2>& rhs) {
			//resize(rhs.size());

			rhs.assign_to(*this, Fcnl_assign<value_type, T2>());
			return *this;
		}

		Map & operator+=(const Vector<value_type> & m) TVMET_CXX_ALWAYS_INLINE
		{		
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Map& operator-=(const Vector<value_type> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}

		Map& operator+=(const Map<Vector<value_type>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		Map& operator-=(const Map<Vector<value_type>> & m) TVMET_CXX_ALWAYS_INLINE
		{
			impl::do_compound_assign(*this, m, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}
		
		template<class E> 
		Map& operator+=(const XprVector<E> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprMatrix<E>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_add_eq<value_type,value_type>());
			return *this;
		}
		template<class E> 
		Map& operator-=(const XprVector<E> & m) TVMET_CXX_ALWAYS_INLINE
		{
			typename XprMatrix<E>::result_type result = m.eval();

			impl::do_compound_assign(*this, result, Fcnl_sub_eq<value_type,value_type>());
			return *this;
		}

		operator Eigen::Matrix<T,Eigen::Dynamic,1> ()
		{

			Eigen::Matrix<T,Eigen::Dynamic,1> M(Size);

			cublasGetVector (Size, sizeof(value_type), m_data,1, M.data(), 1);

			return M;
		}

		void setZero()
		{
			cudaMemset(m_data, 0,size()*sizeof(value_type));
		}

		NoAliasProxy<Map<Vector<T>>> noalias()
		{
			return NoAliasProxy<Map<Vector<T>>>(*this);
		}

	private:
		//template<class Obj, std::size_t LEN> friend class CommaInitializer;

		/** This is a helper for assigning a comma separated initializer
		list. It's equal to Vector& operator=(value_type) which does
		replace it. */
		//Map & assign_value(value_type rhs) {
		//	typedef XprLiteral<value_type> 			expr_type;
		//	*this = XprVector<expr_type>(expr_type(rhs));
		//	return *this;
		//}

	public: // math operators with scalars
		// NOTE: this meaning is clear - element wise ops even if not in ns element_wise
	//	Map & operator+=(value_type) TVMET_CXX_ALWAYS_INLINE;
	//	Map& operator-=(value_type) TVMET_CXX_ALWAYS_INLINE;
	//	Map& operator*=(value_type) TVMET_CXX_ALWAYS_INLINE;
	//	Map& operator/=(value_type) TVMET_CXX_ALWAYS_INLINE;

	//	Map& operator%=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
	//	Map& operator^=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
	//	Map& operator&=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
	//	Map& operator|=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
	//	Map& operator<<=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
	//	Map& operator>>=(std::size_t) TVMET_CXX_ALWAYS_INLINE;

	//public: // math assign operators with vectors
	//	// NOTE: access using the operators in ns element_wise, since that's what is does
	//	template <class T2> Map& M_add_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_sub_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_mul_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_div_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_mod_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_xor_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_and_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_or_eq (const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_shl_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& M_shr_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;

	//public: // math operators with expressions
	//	// NOTE: access using the operators in ns element_wise, since that's what is does
	//	template <class E> Map& M_add_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_sub_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_mul_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_div_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_mod_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_xor_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_and_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_or_eq (const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_shl_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& M_shr_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;

	//public: // aliased math operators with expressions, used with proxy
	//	template <class T2> Map& alias_assign(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& alias_add_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& alias_sub_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& alias_mul_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class T2> Map& alias_div_eq(const Vector<T2>&) TVMET_CXX_ALWAYS_INLINE;

	//	template <class E> Map& alias_assign(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& alias_add_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& alias_sub_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& alias_mul_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;
	//	template <class E> Map& alias_div_eq(const XprVector<E>&) TVMET_CXX_ALWAYS_INLINE;

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
			
			  os << IndentLevel(l++) << "MapVector<"
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


#include <gpumatrix/MapVectorFunctions.h>
#include <gpumatrix/MapVectorOperators.h>


#endif // TVMET_VECTOR_H

// Local Variables:
// mode:C++
// tab-width:8
// End:
