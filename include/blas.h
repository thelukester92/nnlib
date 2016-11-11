#ifndef BLAS_H
#define BLAS_H

namespace nnlib
{

template <typename T>
class BLAS
{};

template<>
class BLAS<double>
{
typedef double T;
public:
	static void copy(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_dcopy(N, X, strideX, Y, strideY);
	}
	
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_daxpy(N, alpha, X, strideX, Y, strideY);
	}
};

}

#endif
