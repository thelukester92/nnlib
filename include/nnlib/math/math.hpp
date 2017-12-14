#ifndef MATH_HPP
#define MATH_HPP

#include "../core/error.hpp"

namespace nnlib
{

/// Utility class for linear algebra and other math operations.
template <typename T = NN_REAL_T>
class Math
{
public:
	/// x[i] = alpha, 0 <= i < n
	static void vFill(T *x, size_t n, size_t s, T alpha);
	
	/// x[i] *= alpha, 0 <= i < n
	static void vScale(T *x, size_t n, size_t s, T alpha);
	
	/// A[i][j] = alpha, 0 <= i < ra, 0 <= j < ca
	static void mFill(T *A, size_t r, size_t c, size_t ld, T alpha);
	
	/// A[i][j] *= alpha, 0 <= i < ra, 0 <= j < ca
	static void mScale(T *A, size_t r, size_t c, size_t ld, T alpha);
	
	/// y = alpha * x + y
	static void vAdd_v(const T *x, size_t n, size_t sx, T *y, size_t sy, T alpha = 1);
	
	/// y = alpha * x + beta * y
	static void vAdd_v(const T *x, size_t n, size_t sx, T *y, size_t sy, T alpha, T beta);
	
	/// A = alpha * x <*> y + beta * A, <*> = outer product
	static void mAdd_vv(const T *x, size_t r, size_t sx, const T *y, size_t c, size_t sy, T *A, size_t lda, T alpha = 1, T beta = 1);
	
	/// y = alpha * A * x^T + beta * y
	static void vAdd_mv(const T *A, size_t ra, size_t ca, size_t lda, const T *x, size_t sx, T *y, size_t sy, T alpha = 1, T beta = 1);
	
	/// y = alpha * A^T * x^T + beta * y
	static void vAdd_mtv(const T *A, size_t ra, size_t ca, size_t lda, const T *x, size_t sx, T *y, size_t sy, T alpha = 1, T beta = 1);
	
	/// B = alpha * A + beta * B
	static void mAdd_m(const T *A, size_t r, size_t c, size_t lda, T *B, size_t ldb, T alpha = 1, T beta = 1);
	
	/// B = alpha * A^T + beta * B
	static void mAdd_mt(const T *A, size_t r, size_t c, size_t lda, T *B, size_t ldb, T alpha = 1, T beta = 1);
	
	/// C = alpha * A * B + beta * C
	static void mAdd_mm(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha = 1, T beta = 1);
	
	/// C = alpha * A^T * B + beta * C
	static void mAdd_mtm(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha = 1, T beta = 1);
	
	/// C = alpha * A * B^T + beta * C
	static void mAdd_mmt(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha = 1, T beta = 1);
};

}

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::Math<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/math.tpp"
	#include "detail/blas.tpp"
	#include "detail/nvblas.tpp"
#endif

#endif
