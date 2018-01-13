#ifndef MATH_MATH_HPP
#define MATH_MATH_HPP

#include "../core/tensor.hpp"
#include "../util/traits.hpp"

namespace nnlib
{

namespace math
{

// MARK: Single tensor operations.

/// Returns the smallest element in x.
template <typename T>
T min(const Tensor<T> &x);

/// Returns the largest element in x.
template <typename T>
T max(const Tensor<T> &x);

/// Returns the sum of elements in x.
template <typename T>
T sum(const Tensor<T> &x);

/// Returns the average of elements in x.
template <typename T>
T mean(const Tensor<T> &x);

/// Returns the variance of elements in x.
template <typename T>
T variance(const Tensor<T> &x, bool sample = false);

/// Scales and shifts the elements in x to lie in [from..to].
template <typename T>
Tensor<T> &normalize(Tensor<T> &x, T from = 0, T to = 1);

/// Scales and shifts the elements in x to lie in [from..to].
template <typename T>
Tensor<T> &&normalize(Tensor<T> &&x, T from = 0, T to = 1);

/// Caps elements in x to lie in [min..max].
template <typename T>
Tensor<T> &clip(Tensor<T> &x, T min, T max);

/// Caps elements in x to lie in [min..max].
template <typename T>
Tensor<T> &&clip(Tensor<T> &&x, T min, T max);

/// Squares each element in x.
template <typename T>
Tensor<T> &square(Tensor<T> &x);

/// Squares each element in x.
template <typename T>
Tensor<T> &&square(Tensor<T> &&x);

// MARK: Tensor-tensor operations.

/// Flattens the dim'th dimension of x into y by summing.
template <typename T>
Tensor<T> &sum(const Tensor<T> &x, Tensor<T> &y, size_t dim);

/// Flattens the dim'th dimension of x into y by summing.
template <typename T>
Tensor<T> &&sum(const Tensor<T> &x, Tensor<T> &&y, size_t dim);

/// Multiplies each element in y by the corresponding element in x.
template <typename T>
Tensor<T> &pointwiseProduct(const Tensor<T> &x, Tensor<T> &y);

/// Multiplies each element in y by the corresponding element in x.
template <typename T>
Tensor<T> &&pointwiseProduct(const Tensor<T> &x, Tensor<T> &&y);

/// Sets each element in z to the product of the corresponding elements in x and y.
template <typename T>
Tensor<T> &pointwiseProduct(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &z);

/// Sets each element in z to the product of the corresponding elements in x and y.
template <typename T>
Tensor<T> &&pointwiseProduct(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &&z);

}

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template NN_REAL_T nnlib::math::min<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &);
    extern template NN_REAL_T nnlib::math::max<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &);
    extern template NN_REAL_T nnlib::math::sum<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &);
    extern template NN_REAL_T nnlib::math::mean<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &);
    extern template NN_REAL_T nnlib::math::variance<NN_REAL_T>(const nnlib::Tensor<NN_REAL_T> &, bool);
    extern template nnlib::Tensor<NN_REAL_T> &nnlib::math::normalize<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template nnlib::Tensor<NN_REAL_T> &&nnlib::math::normalize<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template nnlib::Tensor<NN_REAL_T> &nnlib::math::clip<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &, NN_REAL_T, NN_REAL_T);
    extern template nnlib::Tensor<NN_REAL_T> &&nnlib::math::clip<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &&, NN_REAL_T, NN_REAL_T);
    extern template nnlib::Tensor<NN_REAL_T> &nnlib::math::square<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &);
    extern template nnlib::Tensor<NN_REAL_T> &&nnlib::math::square<NN_REAL_T>(nnlib::Tensor<NN_REAL_T> &&);
    extern template nnlib::Tensor<NN_REAL_T> &nnlib::math::sum(const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &, size_t);
    extern template nnlib::Tensor<NN_REAL_T> &&nnlib::math::sum(const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &&, size_t);
    extern template nnlib::Tensor<NN_REAL_T> &nnlib::math::pointwiseProduct(const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &);
    extern template nnlib::Tensor<NN_REAL_T> &&nnlib::math::pointwiseProduct(const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &&);
    extern template nnlib::Tensor<NN_REAL_T> &nnlib::math::pointwiseProduct(const nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &);
    extern template nnlib::Tensor<NN_REAL_T> &&nnlib::math::pointwiseProduct(const nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &, nnlib::Tensor<NN_REAL_T> &&);
#elif !defined NN_IMPL
    #include "detail/math.tpp"
#endif

#endif
