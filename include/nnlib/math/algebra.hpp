#ifndef MATH_ALGEBRA_HPP
#define MATH_ALGEBRA_HPP

#include "../core/type.hpp"

namespace nnlib
{

template <typename T>
class Tensor;

/// Utility class for linear algebra.
template <typename T = NN_REAL_T>
class Algebra
{
public:
	/// x[i] = alpha, 0 <= i < n
	static void vFill(Tensor<T> &x, T alpha);

	/// x[i] *= alpha, 0 <= i < n
	static void vScale(Tensor<T> &x, T alpha);

	/// A[i][j] = alpha, 0 <= i < ra, 0 <= j < ca
	static void mFill(Tensor<T> &A, T alpha);

	/// A[i][j] *= alpha, 0 <= i < ra, 0 <= j < ca
	static void mScale(Tensor<T> &A, T alpha);

	/// y = alpha * x + y
	static void vAdd_v(const Tensor<T> &x, Tensor<T> &y, T alpha = 1);

	/// y = alpha * x + beta * y
	static void vAdd_v(const Tensor<T> &x, Tensor<T> &y, T alpha, T beta);

	/// A = alpha * x <*> y + beta * A, <*> = outer product
	static void mAdd_vv(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &A, T alpha = 1, T beta = 1);

	/// y = alpha * A * x^T + beta * y
	static void vAdd_mv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y, T alpha = 1, T beta = 1);

	/// y = alpha * A^T * x^T + beta * y
	static void vAdd_mtv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y, T alpha = 1, T beta = 1);

	/// B = alpha * A + beta * B
	static void mAdd_m(const Tensor<T> &A, Tensor<T> &B, T alpha = 1, T beta = 1);

	/// B = alpha * A^T + beta * B
	static void mAdd_mt(const Tensor<T> &A, Tensor<T> &B, T alpha = 1, T beta = 1);

	/// C = alpha * A * B + beta * C
	static void mAdd_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, T alpha = 1, T beta = 1);

	/// C = alpha * A^T * B + beta * C
	static void mAdd_mtm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, T alpha = 1, T beta = 1);

	/// C = alpha * A * B^T + beta * C
	static void mAdd_mmt(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, T alpha = 1, T beta = 1);
};

}

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::Algebra<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/algebra.tpp"
	#include "detail/algebra_blas.tpp"
	#include "detail/algebra_nvblas.tpp"
#endif

#endif
