#ifndef BLAS_H
#define BLAS_H

#ifdef APPLE
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif

namespace nnlib
{

/// Fallback methods to mimic BLAS for non-floating types.
/// \todo gemm in the fallback

template <typename T>
class BLAS
{
public:
	static void set(size_t N, T alpha, T *X, size_t strideX)
	{
		for(size_t i = 0; i < N; ++i)
			X[i * strideX] = alpha;
	}
	
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
	
	static void gemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, size_t rows, size_t cols, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		if(transA == CblasNoTrans)
		{
			for(size_t i = 0; i < rows; ++i)
			{
				Y[i * strideY] *= beta;
				for(size_t j = 0; j < cols; ++j)
					Y[i * strideY] += alpha * A[i * lda + j] * X[j * strideX];
			}
		}
		else
		{
			for(size_t i = 0; i < cols; ++i)
			{
				Y[i * strideY] *= beta;
				for(size_t j = 0; j < rows; ++j)
					Y[i * strideY] += alpha * A[j * lda + i] * X[j * strideX];
			}
		}
	}
};

/// Single-precision BLAS.
template<>
class BLAS<float>
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
	
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_saxpy(N, alpha, X, strideX, Y, strideY);
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
class BLAS<double>
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
	
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_daxpy(N, alpha, X, strideX, Y, strideY);
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
