#ifndef MATH_ALGEBRA_HPP
#define MATH_ALGEBRA_HPP

#include "../core/type.hpp"
#include "../util/traits.hpp"

namespace nnlib
{

template <typename T>
class Tensor;

namespace math
{

/// x[i] = alpha, 0 <= i < n
template <typename T>
void vFill(Tensor<T> &x, typename traits::Identity<T>::type alpha);

/// x[i] = alpha, 0 <= i < n
template <typename T>
void vFill(Tensor<T> &&x, typename traits::Identity<T>::type alpha);

/// x[i] *= alpha, 0 <= i < n
template <typename T>
void vScale(Tensor<T> &x, typename traits::Identity<T>::type alpha);

/// x[i] *= alpha, 0 <= i < n
template <typename T>
void vScale(Tensor<T> &&x, typename traits::Identity<T>::type alpha);

/// A[i][j] = alpha, 0 <= i < ra, 0 <= j < ca
template <typename T>
void mFill(Tensor<T> &A, typename traits::Identity<T>::type alpha);

/// A[i][j] = alpha, 0 <= i < ra, 0 <= j < ca
template <typename T>
void mFill(Tensor<T> &&A, typename traits::Identity<T>::type alpha);

/// A[i][j] *= alpha, 0 <= i < ra, 0 <= j < ca
template <typename T>
void mScale(Tensor<T> &A, typename traits::Identity<T>::type alpha);

/// A[i][j] *= alpha, 0 <= i < ra, 0 <= j < ca
template <typename T>
void mScale(Tensor<T> &&A, typename traits::Identity<T>::type alpha);

/// y = alpha * x + y
template <typename T>
void vAdd_v(const Tensor<T> &x, Tensor<T> &y, typename traits::Identity<T>::type alpha = 1);

/// y = alpha * x + y
template <typename T>
void vAdd_v(const Tensor<T> &x, Tensor<T> &&y, typename traits::Identity<T>::type alpha = 1);

/// y = alpha * x + beta * y
template <typename T>
void vAdd_v(const Tensor<T> &x, Tensor<T> &y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta);

/// y = alpha * x + beta * y
template <typename T>
void vAdd_v(const Tensor<T> &x, Tensor<T> &&y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta);

/// A = alpha * x <*> y + beta * A, <*> = outer product
template <typename T>
void mAdd_vv(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &A, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// A = alpha * x <*> y + beta * A, <*> = outer product
template <typename T>
void mAdd_vv(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &&A, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// y = alpha * A * x^T + beta * y
template <typename T>
void vAdd_mv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// y = alpha * A * x^T + beta * y
template <typename T>
void vAdd_mv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &&y, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// y = alpha * A^T * x^T + beta * y
template <typename T>
void vAdd_mtv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// y = alpha * A^T * x^T + beta * y
template <typename T>
void vAdd_mtv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &&y, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// B = alpha * A + beta * B
template <typename T>
void mAdd_m(const Tensor<T> &A, Tensor<T> &B, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// B = alpha * A + beta * B
template <typename T>
void mAdd_m(const Tensor<T> &A, Tensor<T> &&B, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// B = alpha * A^T + beta * B
template <typename T>
void mAdd_mt(const Tensor<T> &A, Tensor<T> &B, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// B = alpha * A^T + beta * B
template <typename T>
void mAdd_mt(const Tensor<T> &A, Tensor<T> &&B, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// C = alpha * A * B + beta * C
template <typename T>
void mAdd_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// C = alpha * A * B + beta * C
template <typename T>
void mAdd_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &&C, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// C = alpha * A^T * B + beta * C
template <typename T>
void mAdd_mtm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// C = alpha * A^T * B + beta * C
template <typename T>
void mAdd_mtm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &&C, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// C = alpha * A * B^T + beta * C
template <typename T>
void mAdd_mmt(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

/// C = alpha * A * B^T + beta * C
template <typename T>
void mAdd_mmt(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &&C, typename traits::Identity<T>::type alpha = 1, typename traits::Identity<T>::type beta = 1);

} // namespace math

} // namespace nnlib

#if defined NN_REAL_T && !defined NN_IMPL
    extern template void nnlib::math::vFill<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
    extern template void nnlib::math::vFill<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
    extern template void nnlib::math::vScale<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
    extern template void nnlib::math::vScale<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
    extern template void nnlib::math::mFill<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
    extern template void nnlib::math::mFill<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
    extern template void nnlib::math::mScale<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
    extern template void nnlib::math::mScale<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
    extern template void nnlib::math::vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T);
    extern template void nnlib::math::vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T);
    extern template void nnlib::math::vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_vv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_vv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::vAdd_mv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::vAdd_mv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::vAdd_mtv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::vAdd_mtv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_m<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_m<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_mt<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_mt<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_mm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_mm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_mtm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_mtm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_mmt<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template void nnlib::math::mAdd_mmt<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
#elif !defined NN_IMPL
    #include "detail/algebra.tpp"
    #include "detail/algebra_blas.tpp"
    #include "detail/algebra_nvblas.tpp"
#endif

#endif
