#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/math/algebra.hpp"
#include "nnlib/math/detail/algebra.tpp"
#include "nnlib/math/detail/algebra_blas.tpp"
#include "nnlib/math/detail/algebra_nvblas.tpp"

template void nnlib::math::vFill<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template void nnlib::math::vFill<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template void nnlib::math::vScale<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template void nnlib::math::vScale<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template void nnlib::math::mFill<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template void nnlib::math::mFill<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template void nnlib::math::mScale<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template void nnlib::math::mScale<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template void nnlib::math::vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T);
template void nnlib::math::vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T);
template void nnlib::math::vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::vAdd_v<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_vv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_vv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void nnlib::math::vAdd_mv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::vAdd_mv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void nnlib::math::vAdd_mtv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::vAdd_mtv<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_m<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_m<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_mt<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_mt<NN_REAL_T>(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_mm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_mm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_mtm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_mtm<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_mmt<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template void nnlib::math::mAdd_mmt<NN_REAL_T>(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);

#endif
