#ifndef ALGEBRA_H
#define ALGEBRA_H

#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#endif

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
		// return cblas_ddot(N, x, strideX, y, strideY);
	}
};

template <>
class Algebra<float>
{
typedef float T;
public:
	static T dot(size_t N, T *x, size_t strideX, T *y, size_t strideY)
	{
		// return cblas_sdot(N, x, strideX, y, strideY);
	}
};

}

#endif
