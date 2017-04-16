#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <Accelerate/Accelerate.h>

namespace nnlib
{

template <typename T>
class Algebra
{};

template <>
class Algebra<double>
{
typedef double T;
public:
	static T dot(size_t N, T *x, size_t strideX, T *y, size_t strideY)
	{
		return cblas_ddot(N, x, strideX, y, strideY);
	}
};

template <>
class Algebra<float>
{
typedef float T;
public:
	static T dot(size_t N, T *x, size_t strideX, T *y, size_t strideY)
	{
		return cblas_sdot(N, x, strideX, y, strideY);
	}
};

template <>
class Algebra<size_t>
{
typedef size_t T;
public:
	static T dot(size_t N, T *x, size_t strideX, T *y, size_t strideY)
	{
		T sum = 0;
		for(size_t i = 0; i < N; ++i)
		{
			sum += x[i] * y[i];
			x += strideX;
			y += strideY;
		}
		return sum;
	}
};

}

#endif
