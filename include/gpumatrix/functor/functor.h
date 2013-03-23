#ifndef FUNCTOR_H_
#define FUNCTOR_H_


struct saxpy_functor
{
	const float a;

	saxpy_functor(float _a) : a(_a) {}

	__host__ __device__
		float operator()(const float& x, const float& y) const { 
			return a * x + y;
		}
};

template<typename T>
struct scal_functor
{
	const T & a;

	saxpy_functor(T & _a) : a(_a) {}

	__host__ __device__
	T operator()(const T & x) const 
	{ 
			return a * x;
	}
};

#endif