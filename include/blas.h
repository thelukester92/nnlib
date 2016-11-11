#ifndef BLAS_H
#define BLAS_H

namespace nnlib
{

/// Fallback methods to mimic BLAS for non-floating types.
/// \todo determine if this should be renamed.
template <typename T>
class BLAS
{
public:
	static void copy(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		for(size_t i = 0; i < N; ++i)
			Y[i * strideY] = X[i * strideX];
	}
	
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		for(size_t i = 0; i < N; ++i)
			Y[i * strideY] += alpha * X[i * strideX];
	}
};

/// Double-precision BLAS.
template<>
class BLAS<double>
{
typedef double T;
public:
	static void copy(size_t n, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_dcopy(n, X, strideX, Y, strideY);
	}
	
	static void axpy(size_t n, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_daxpy(n, alpha, X, strideX, Y, strideY);
	}
	
	static void gemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, size_t rows, size_t cols, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_dgemv(order, transA, rows, cols, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
};

}

#endif
