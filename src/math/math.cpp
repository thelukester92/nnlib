#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/math/math.hpp"
#include "nnlib/math/detail/math.tpp"

template NN_REAL_T nnlib::math::min<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &);
template NN_REAL_T nnlib::math::max<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &);
template NN_REAL_T nnlib::math::sum<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &);
template NN_REAL_T nnlib::math::mean<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &);
template NN_REAL_T nnlib::math::variance<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &, bool);
template nnlib::Tensor<NN_REAL_T> &nnlib::math::clip<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template nnlib::Tensor<NN_REAL_T> &&nnlib::math::clip<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template nnlib::Tensor<NN_REAL_T> &nnlib::math::square<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &);
template nnlib::Tensor<NN_REAL_T> &&nnlib::math::square<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &&);
template nnlib::Tensor<NN_REAL_T> &nnlib::math::sum(const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &, size_t);
template nnlib::Tensor<NN_REAL_T> &&nnlib::math::sum(const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &&, size_t);
template nnlib::Tensor<NN_REAL_T> &nnlib::math::pointwiseProduct(const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &);
template nnlib::Tensor<NN_REAL_T> &&nnlib::math::pointwiseProduct(const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &&);
template nnlib::Tensor<NN_REAL_T> &nnlib::math::pointwiseProduct(const nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &);
template nnlib::Tensor<NN_REAL_T> &&nnlib::math::pointwiseProduct(const nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &&);

#endif
