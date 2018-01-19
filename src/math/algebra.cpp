#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/math/algebra.hpp"
#include "nnlib/math/detail/algebra.tpp"
#include "nnlib/math/detail/algebra_blas.tpp"
#include "nnlib/math/detail/algebra_nvblas.tpp"

namespace nnlib
{

namespace math
{

template void vFill<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template void vFill<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template void vScale<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template void vScale<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template void mFill<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template void mFill<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template void mScale<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template void mScale<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template void vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T);
template void vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T);
template void vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void mAdd_vv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void mAdd_vv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void vAdd_mv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void vAdd_mv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void vAdd_mtv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void vAdd_mtv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void mAdd_m<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void mAdd_m<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void mAdd_mt<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void mAdd_mt<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void mAdd_mm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void mAdd_mm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void mAdd_mtm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void mAdd_mtm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void mAdd_mmt<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void mAdd_mmt<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);

} // namespace math

} // namespace nnlib

#endif
