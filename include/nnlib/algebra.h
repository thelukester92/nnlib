#ifndef ALGEBRA_H
#define ALGEBRA_H

#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#else
	extern "C"
	{
		#include <cblas.h>
	};
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
	static Algebra &instance()
	{
		static Algebra algebra;
		return algebra;
	}
	
	Algebra(const Algebra &) = delete;
	void operator=(const Algebra &) = delete;
	
	/// X.fill(alpha)
	void set(size_t N, T alpha, T *X, size_t strideX)
	{
		// catlas_sset(N, alpha, X, strideX);
		for(size_t i = 0; i < N; ++i)
			X[i * strideX] = alpha;
	}
	
	/// Y = X
	void copy(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_scopy(N, X, strideX, Y, strideY);
	}
	
	/// X *= scalar
	void scal(size_t N, T scalar, T *X, size_t strideX)
	{
		cblas_sscal(N, scalar, X, strideX);
	}
	
	/// Y += alpha * X
	void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_saxpy(N, alpha, X, strideX, Y, strideY);
	}
	
	/// Swap contents of X and Y.
	void swap(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_sswap(N, X, strideX, Y, strideY);
	}
	
	/// Outer product; A += alpha * X * Y'
	void ger(CBLAS_ORDER order, size_t rows, size_t cols, T alpha, T *X, size_t strideX, T *Y, size_t strideY, T *A, size_t lda)
	{
		cblas_sger(order, rows, cols, alpha, X, strideX, Y, strideY, A, lda);
	}
	
	/// Matrix-vector multiplication.
	void gemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, size_t rows, size_t cols, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_sgemv(order, transA, rows, cols, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
	
	/// Matrix-matrix multiplication.
	void gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, size_t rows, size_t cols, size_t inner, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_sgemm(order, transA, transB, rows, cols, inner, alpha, A, lda, B, ldb, beta, C, ldc);
	}

private:
	Algebra() {}
};

/// Double-precision BLAS.
template<>
class Algebra<double>
{
typedef double T;
public:
	static Algebra &instance()
	{
		static Algebra algebra;
		return algebra;
	}
	
	Algebra(const Algebra &) = delete;
	void operator=(const Algebra &) = delete;
	
	void set(size_t N, T alpha, T *X, size_t strideX)
	{
		// catlas_dset(N, alpha, X, strideX);
		for(size_t i = 0; i < N; ++i)
			X[i * strideX] = alpha;
	}
	
	void copy(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_dcopy(N, X, strideX, Y, strideY);
	}
	
	void scal(size_t N, T scalar, T *X, size_t strideX)
	{
		cblas_dscal(N, scalar, X, strideX);
	}
	
	void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_daxpy(N, alpha, X, strideX, Y, strideY);
	}
	
	void swap(size_t N, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_dswap(N, X, strideX, Y, strideY);
	}
	
	void ger(CBLAS_ORDER order, size_t rows, size_t cols, T alpha, T *X, size_t strideX, T *Y, size_t strideY, T *A, size_t lda)
	{
		cblas_dger(order, rows, cols, alpha, X, strideX, Y, strideY, A, lda);
	}
	
	void gemv(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, size_t rows, size_t cols, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_dgemv(order, transA, rows, cols, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
	
	void gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, size_t rows, size_t cols, size_t inner, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_dgemm(order, transA, transB, rows, cols, inner, alpha, A, lda, B, ldb, beta, C, ldc);
	}

private:
	Algebra() {}
};

}

#endif
