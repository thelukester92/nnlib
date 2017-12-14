#ifndef MATH_MATH_BLAS_HPP
#define MATH_MATH_BLAS_HPP

#ifdef NN_ACCEL
	#ifdef __APPLE__
		#include <Accelerate/Accelerate.h>
	#else
		#include <cblas.h>
	#endif
#endif

#include "math_base.hpp"

namespace nnlib
{

/// BLAS-accelerated math.
template <typename T>
class MathBLAS : public MathBase<T>
{};

#ifdef NN_ACCEL

/// Single-precision BLAS-accelerated math.
template <>
class MathBLAS<float> : public MathBase<float>
{
public:
	using T = float;
	
	// MARK: Vector/scalar operations
	
	#ifdef __APPLE__
		/// x[i] = alpha, 0 <= i < n
		/// \note Only available when using Accelerate on OS X.
		static void vFill(T *x, size_t n, size_t s, T alpha)
		{
			catlas_sset(n, alpha, x, s);
		}
	#endif
	
	/// x[i] *= alpha, 0 <= i < n
	static void vScale(T *x, size_t n, size_t s, T alpha)
	{
		cblas_sscal(n, alpha, x, s);
	}
	
	// MARK: Matrix/scalar operations
	
	#ifdef __APPLE__
		/// A[i][j] = alpha, 0 <= i < ra, 0 <= j < ca
		/// \note Only available when using Accelerate on OS X.
		static void mFill(T *A, size_t r, size_t c, size_t ld, T alpha)
		{
			for(size_t i = 0; i < r; ++i, A += ld)
				catlas_sset(c, alpha, A, 1);
		}
	#endif
	
	/// A[i][j] *= alpha, 0 <= i < ra, 0 <= j < ca
	static void mScale(T *A, size_t r, size_t c, size_t ld, T alpha)
	{
		for(size_t i = 0; i < r; ++i, A += ld)
			cblas_sscal(c, alpha, A, 1);
	}
	
	// MARK: Vector/Vector operations
	
	/// y = alpha * x + y
	static void vAdd_v(
		const T *x, size_t n, size_t sx,
		T *y, size_t sy,
		T alpha = 1
	)
	{
		cblas_saxpy(n, alpha, x, sx, y, sy);
	}
	
	/// y = alpha * x + beta * y
	static void vAdd_v(
		const T *x, size_t n, size_t sx,
		T *y, size_t sy,
		T alpha, T beta
	)
	{
		#ifdef __APPLE__
			catlas_saxpby(n, alpha, x, sx, beta, y, sy);
		#else
			cblas_saxpby(n, alpha, x, sx, beta, y, sy);
		#endif
	}
	
	// MARK: Matrix/Vector operations
	
	/// A = alpha * x <*> y + beta * A, <*> = outer product
	static void mAdd_vv(
		const T *x, size_t r, size_t sx,
		const T *y, size_t c, size_t sy,
		T *A, size_t lda,
		T alpha = 1, T beta = 1
	)
	{
		if(beta != 1.0f)
			mScale(A, r, c, lda, beta);
		cblas_sger(CblasRowMajor, r, c, alpha, x, sx, y, sy, A, lda);
	}
	
	/// y = alpha * A * x^T + beta * y
	static void vAdd_mv(
		const T *A, size_t ra, size_t ca, size_t lda,
		const T *x, size_t sx,
		T *y, size_t sy,
		T alpha = 1, T beta = 1
	)
	{
		cblas_sgemv(CblasRowMajor, CblasNoTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
	}
	
	/// y = alpha * A^T * x^T + beta * y
	static void vAdd_mtv(
		const T *A, size_t ra, size_t ca, size_t lda,
		const T *x, size_t sx,
		T *y, size_t sy,
		T alpha = 1, T beta = 1
	)
	{
		cblas_sgemv(CblasRowMajor, CblasTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
	}
	
	// MARK: Matrix/Matrix operations
	
	/// C = alpha * A * B + beta * C
	static void mAdd_mm(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 1
	)
	{
		cblas_sgemm(
			CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	/// C = alpha * A^T * B + beta * C
	static void mAdd_mtm(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 1
	)
	{
		cblas_sgemm(
			CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	/// C = alpha * A * B^T + beta * C
	static void mAdd_mmt(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 1
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
	
	// MARK: Vector/scalar operations
	
	#ifdef __APPLE__
		/// x[i] = alpha, 0 <= i < n
		/// \note Only available when using Accelerate on OS X.
		static void vFill(T *x, size_t n, size_t s, T alpha)
		{
			catlas_dset(n, alpha, x, s);
		}
	#endif
	
	/// x[i] *= alpha, 0 <= i < n
	static void vScale(T *x, size_t n, size_t s, T alpha)
	{
		cblas_dscal(n, alpha, x, s);
	}
	
	// MARK: Matrix/scalar operations
	
	#ifdef __APPLE__
		/// A[i][j] = alpha, 0 <= i < ra, 0 <= j < ca
		/// \note Only available when using Accelerate on OS X.
		static void mFill(T *A, size_t r, size_t c, size_t ld, T alpha)
		{
			for(size_t i = 0; i < r; ++i, A += ld)
				catlas_dset(c, alpha, A, 1);
		}
	#endif
	
	/// A[i][j] *= alpha, 0 <= i < ra, 0 <= j < ca
	static void mScale(T *A, size_t r, size_t c, size_t ld, T alpha)
	{
		for(size_t i = 0; i < r; ++i, A += ld)
			cblas_dscal(c, alpha, A, 1);
	}
	
	// MARK: Vector/Vector operations
	
	/// y = alpha * x + y
	static void vAdd_v(
		const T *x, size_t n, size_t sx,
		T *y, size_t sy,
		T alpha = 1
	)
	{
		cblas_daxpy(n, alpha, x, sx, y, sy);
	}
	
	/// y = alpha * x + beta * y
	static void vAdd_v(
		const T *x, size_t n, size_t sx,
		T *y, size_t sy,
		T alpha, T beta
	)
	{
		#ifdef __APPLE__
			catlas_daxpby(n, alpha, x, sx, beta, y, sy);
		#else
			cblas_daxpby(n, alpha, x, sx, beta, y, sy);
		#endif
	}
	
	// MARK: Matrix/Vector operations
	
	/// A = alpha * x <*> y + beta * A, <*> = outer product
	static void mAdd_vv(
		const T *x, size_t r, size_t sx,
		const T *y, size_t c, size_t sy,
		T *A, size_t lda,
		T alpha = 1, T beta = 1
	)
	{
		if(beta != 1)
			mScale(A, r, c, lda, beta);
		cblas_dger(CblasRowMajor, r, c, alpha, x, sx, y, sy, A, lda);
	}
	
	/// y = alpha * A * x^T + beta * y
	static void vAdd_mv(
		const T *A, size_t ra, size_t ca, size_t lda,
		const T *x, size_t sx,
		T *y, size_t sy,
		T alpha = 1, T beta = 1
	)
	{
		cblas_dgemv(CblasRowMajor, CblasNoTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
	}
	
	/// y = alpha * A^T * x^T + beta * y
	static void vAdd_mtv(
		const T *A, size_t ra, size_t ca, size_t lda,
		const T *x, size_t sx,
		T *y, size_t sy,
		T alpha = 1, T beta = 1
	)
	{
		cblas_dgemv(CblasRowMajor, CblasTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
	}
	
	/// y = alpha * A * x^T + beta * y
	static void mAdd_mv(
		const T *A, size_t ra, size_t ca, size_t lda,
		const T *x, size_t sx,
		T *y, size_t sy,
		T alpha = 1, T beta = 1
	)
	{
		cblas_dgemv(CblasRowMajor, CblasNoTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
	}
	
	/// y = alpha * A^T * x^T + beta * y
	static void mAdd_mtv(
		const T *A, size_t ra, size_t ca, size_t lda,
		const T *x, size_t sx,
		T *y, size_t sy,
		T alpha = 1, T beta = 1
	)
	{
		cblas_dgemv(CblasRowMajor, CblasTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
	}
	
	// MARK: Matrix/Matrix operations
	
	/// C = alpha * A * B + beta * C
	static void mAdd_mm(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 1
	)
	{
		cblas_dgemm(
			CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	/// C = alpha * A^T * B + beta * C
	static void mAdd_mtm(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 1
	)
	{
		cblas_dgemm(
			CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha,
			const_cast<T *>(A), lda,
			const_cast<T *>(B), ldb,
			beta, C, ldc
		);
	}
	
	/// C = alpha * A * B^T + beta * C
	static void mAdd_mmt(
		size_t M, size_t N, size_t K,
		const T *A, size_t lda,
		const T *B, size_t ldb,
		T *C, size_t ldc,
		T alpha = 1, T beta = 1
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

#endif

}

#endif
