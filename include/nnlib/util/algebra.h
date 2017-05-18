#ifndef ALGEBRA_H
#define ALGEBRA_H

#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif

#include "../error.h"

namespace nnlib
{

/// BLAS wrapper to move type out to a template parameter.
template <typename T>
class BLAS
{};

/// Double-precision BLAS.
template <>
class BLAS<double>
{
using T = double;
public:
	/// Matrix x Matrix
	static void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, size_t M, size_t N, size_t K, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_dgemm(o, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	
	/// Matrix x Vector
	static void gemv(CBLAS_ORDER o, CBLAS_TRANSPOSE trans, size_t M, size_t N, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_dgemv(o, trans, M, N, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
	
	/// Vector x Vector (outer product)
	static void ger(CBLAS_ORDER o, size_t M, size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY, T *A, size_t lda)
	{
		cblas_dger(o, M, N, alpha, X, strideX, Y, strideY, A, lda);
	}
	
	/// Vector + Vector (scaled)
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_daxpy(N, alpha, X, strideX, Y, strideY);
	}
};

/// Single-precision BLAS.
template <>
class BLAS<float>
{
using T = float;
public:
	/// Matrix x Matrix
	static void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, size_t M, size_t N, size_t K, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_sgemm(o, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	
	/// Matrix x Vector
	static void gemv(CBLAS_ORDER o, CBLAS_TRANSPOSE trans, size_t M, size_t N, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_sgemv(o, trans, M, N, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
	
	/// Vector x Vector (outer product)
	static void ger(CBLAS_ORDER o, size_t M, size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY, T *A, size_t lda)
	{
		cblas_sger(o, M, N, alpha, X, strideX, Y, strideY, A, lda);
	}
	
	/// Vector + Vector (scaled)
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_saxpy(N, alpha, X, strideX, Y, strideY);
	}
};

/// Abstracted BLAS with runtime checks.
template <typename T>
class Algebra
{
public:
	/// C = alpha * A * B + beta * C
	static void multiplyMM(
		const T *A, size_t rowsA, size_t colsA, size_t lda,
		const T *B, size_t rowsB, size_t colsB, size_t ldb,
		T *C, size_t rowsC, size_t colsC, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		NNAssert(colsA == rowsB && rowsA == rowsC && colsB == colsC, "Incompatible operands!");
		BLAS<T>::gemm(
			CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, colsB, colsA, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	/// C = alpha * A^T * B + beta * C
	static void multiplyMTM(
		const T *A, size_t rowsA, size_t colsA, size_t lda,
		const T *B, size_t rowsB, size_t colsB, size_t ldb,
		T *C, size_t rowsC, size_t colsC, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		NNAssert(rowsA == rowsB && colsA == rowsC && colsB == colsC, "Incompatible operands!");
		BLAS<T>::gemm(
			CblasRowMajor, CblasTrans, CblasNoTrans, colsA, colsB, rowsA, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	/// C = alpha * A * B^T + beta * C
	static void multiplyMMT(
		const T *A, size_t rowsA, size_t colsA, size_t lda,
		const T *B, size_t rowsB, size_t colsB, size_t ldb,
		T *C, size_t rowsC, size_t colsC, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		NNAssert(colsA == colsB && rowsA == rowsC && rowsB == colsC, "Incompatible operands!");
		BLAS<T>::gemm(
			CblasRowMajor, CblasNoTrans, CblasTrans, rowsA, rowsB, colsA, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	/// C = alpha * A^T * B^T + beta * C
	static void multiplyMTMT(
		const T *A, size_t rowsA, size_t colsA, size_t lda,
		const T *B, size_t rowsB, size_t colsB, size_t ldb,
		T *C, size_t rowsC, size_t colsC, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		NNAssert(rowsA == colsB && colsA == rowsC && rowsB == colsC, "Incompatible operands!");
		BLAS<T>::gemm(
			CblasRowMajor, CblasTrans, CblasTrans, colsA, rowsB, rowsA, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	/// y = alpha * A * x + beta * y
	static void multiplyMV(
		const T *A, size_t rowsA, size_t colsA, size_t lda,
		const T *x, size_t sizeX, size_t strideX,
		T *y, size_t sizeY, size_t strideY,
		T alpha = 1, T beta = 0
	)
	{
		NNAssert(colsA == sizeX && rowsA == sizeY, "Incompatible operands!");
		BLAS<T>::gemv(
			CblasRowMajor, CblasNoTrans, rowsA, colsA, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(x), strideX,
			beta, y, strideY
		);
	}
	
	/// y = alpha * A^T * x + beta * y
	static void multiplyMTV(
		const T *A, size_t rowsA, size_t colsA, size_t lda,
		const T *x, size_t sizeX, size_t strideX,
		T *y, size_t sizeY, size_t strideY,
		T alpha = 1, T beta = 0
	)
	{
		NNAssert(rowsA == sizeX && colsA == sizeY, "Incompatible operands!");
		BLAS<T>::gemv(
			CblasRowMajor, CblasTrans, rowsA, colsA, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(x), strideX,
			beta, y, strideY
		);
	}
	
	/// A = alpha * x^T * y + A
	static void multiplyVTV(
		const T *x, size_t sizeX, size_t strideX,
		const T *y, size_t sizeY, size_t strideY,
		T *A, size_t rowsA, size_t colsA, size_t lda,
		T alpha = 1
	)
	{
		NNAssert(rowsA == sizeX && colsA == sizeY, "Incompatible operands!");
		BLAS<T>::ger(
			CblasRowMajor, rowsA, colsA, alpha,
			const_cast<T *>(x), strideX,
			const_cast<T *>(y), strideY,
			A, lda
		);
	}
	
	/// y = alpha * x + y
	static void addVV(
		const T *x, size_t sizeX, size_t strideX,
		T *y, size_t sizeY, size_t strideY,
		T alpha = 1
	)
	{
		NNAssert(sizeX == sizeY, "Incompatible operands!");
		BLAS<T>::axpy(
			sizeX, alpha,
			const_cast<T *>(x), strideX,
			y, strideY
		);
	}
};

}

#endif
