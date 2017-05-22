#ifndef MATH_BLAS_H
#define MATH_BLAS_H

#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif

#include "math_base.h"

namespace nnlib
{

/// BLAS-accelerated math.
template <typename T>
class MathBLAS : public MathBase<T>
{};

/// Single-precision BLAS-accelerated math.
template <>
class MathBLAS<float> : public MathBase<float>
{
public:
	using T = float;
	
	static void mAdd_mm(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		cblas_sgemm(
			CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	static void mAdd_mtm(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		cblas_sgemm(
			CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	static void mAdd_mmt(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		cblas_sgemm(
			CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
};

/// Double-precision BLAS-accelerated math.
template <>
class MathBLAS<double> : public MathBase<double>
{
public:
	using T = double;
	
	static void mAdd_mm(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		cblas_dgemm(
			CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	static void mAdd_mtm(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		cblas_dgemm(
			CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	static void mAdd_mmt(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 0
	)
	{
		cblas_dgemm(
			CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
};

}

#endif
