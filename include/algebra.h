#ifndef ALGEBRA_H
#define ALGEBRA_H

#ifdef APPLE
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif

namespace nnlib
{

/// BLAS-style methods for type T.
template <typename T>
class Algebra
{};

/// Single-precision BLAS.
template<>
class Algebra<float>
{
typedef float T;
public:
	static void set(size_t N, T alpha, T *X, size_t strideX)
	{
		catlas_sset(N, alpha, X, strideX);
	}
	
	static void copy(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_scopy(N, X, strideX, Y, strideY);
	}
	
	static void scal(size_t N, T scalar, T *X, size_t strideX)
	{
		cblas_sscal(N, scalar, X, strideX);
	}
	
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_saxpy(N, alpha, X, strideX, Y, strideY);
	}
	
	static void swap(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_sswap(N, X, strideX, Y, strideY);
	}
	
	static void gemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, size_t rows, size_t cols, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_sgemv(order, transA, rows, cols, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
	
	static void gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, size_t rows, size_t cols, size_t inner, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_sgemm(order, transA, transB, rows, cols, inner, alpha, A, lda, B, ldb, beta, C, ldc);
	}
};

/// Double-precision BLAS.
template<>
class Algebra<double>
{
typedef double T;
public:
	static void set(size_t N, T alpha, T *X, size_t strideX)
	{
		catlas_dset(N, alpha, X, strideX);
	}
	
	static void copy(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_dcopy(N, X, strideX, Y, strideY);
	}
	
	static void scal(size_t N, T scalar, T *X, size_t strideX)
	{
		cblas_dscal(N, scalar, X, strideX);
	}
	
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_daxpy(N, alpha, X, strideX, Y, strideY);
	}
	
	static void swap(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_dswap(N, X, strideX, Y, strideY);
	}
	
	static void gemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, size_t rows, size_t cols, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_dgemv(order, transA, rows, cols, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
	
	static void gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, size_t rows, size_t cols, size_t inner, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_dgemm(order, transA, transB, rows, cols, inner, alpha, A, lda, B, ldb, beta, C, ldc);
	}
};

}

#endif
