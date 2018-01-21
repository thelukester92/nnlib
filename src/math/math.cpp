#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/math/math.hpp"
#include "nnlib/math/detail/math.tpp"

namespace nnlib
{

namespace math
{

template NN_REAL_T min<NN_REAL_T>(const Tensor<NN_REAL_T> &);
template NN_REAL_T max<NN_REAL_T>(const Tensor<NN_REAL_T> &);
template NN_REAL_T sum<NN_REAL_T>(const Tensor<NN_REAL_T> &);
template NN_REAL_T mean<NN_REAL_T>(const Tensor<NN_REAL_T> &);
template NN_REAL_T variance<NN_REAL_T>(const Tensor<NN_REAL_T> &, bool);
template Tensor<NN_REAL_T> &fill(Tensor<NN_REAL_T> &x, NN_REAL_T value);
template Tensor<NN_REAL_T> &&fill(Tensor<NN_REAL_T> &&x, NN_REAL_T value);
template Tensor<NN_REAL_T> &scale(Tensor<NN_REAL_T> &x, NN_REAL_T value);
template Tensor<NN_REAL_T> &&scale(Tensor<NN_REAL_T> &&x, NN_REAL_T value);
template Tensor<NN_REAL_T> &add(Tensor<NN_REAL_T> &x, NN_REAL_T value);
template Tensor<NN_REAL_T> &&add(Tensor<NN_REAL_T> &&x, NN_REAL_T value);
template Tensor<NN_REAL_T> &diminish(Tensor<NN_REAL_T> &x, NN_REAL_T value);
template Tensor<NN_REAL_T> &&diminish(Tensor<NN_REAL_T> &&x, NN_REAL_T value);
template Tensor<NN_REAL_T> &normalize<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &&normalize<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &clip<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &&clip<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &square<NN_REAL_T>(Tensor<NN_REAL_T> &);
template Tensor<NN_REAL_T> &&square<NN_REAL_T>(Tensor<NN_REAL_T> &&);
template Tensor<NN_REAL_T> &rand<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &&rand<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &randn<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &&randn<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &randn<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &&randn<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T, NN_REAL_T);
template Tensor<NN_REAL_T> &bernoulli<NN_REAL_T>(Tensor<NN_REAL_T> &, NN_REAL_T);
template Tensor<NN_REAL_T> &&bernoulli<NN_REAL_T>(Tensor<NN_REAL_T> &&, NN_REAL_T);
template Tensor<NN_REAL_T> &sum(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &, size_t);
template Tensor<NN_REAL_T> &&sum(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&, size_t);
template Tensor<NN_REAL_T> &pointwiseProduct(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &);
template Tensor<NN_REAL_T> &&pointwiseProduct(const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&);
template Tensor<NN_REAL_T> &pointwiseProduct(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &);
template Tensor<NN_REAL_T> &&pointwiseProduct(const Tensor<NN_REAL_T> &, const Tensor<NN_REAL_T> &, Tensor<NN_REAL_T> &&);

} // namespace math

} // namespace nnlib

#endif
