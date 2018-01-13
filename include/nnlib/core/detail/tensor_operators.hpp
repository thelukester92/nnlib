#ifndef CORE_TENSOR_OPERATORS_HPP
#define CORE_TENSOR_OPERATORS_HPP

#include "../tensor.hpp"
#include "nnlib/util/traits.hpp"

template <typename T>
std::ostream &operator<<(std::ostream &out, const nnlib::Tensor<T> &t);

template <typename T>
nnlib::Tensor<T> &operator+=(nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs);

template <typename T>
nnlib::Tensor<T> operator+(const nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs);

template <typename T>
nnlib::Tensor<T> &operator-=(nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs);

template <typename T>
nnlib::Tensor<T> operator-(const nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs);

template <typename T>
nnlib::Tensor<T> &operator*=(nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs);

template <typename T>
nnlib::Tensor<T> operator*(const nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs);

template <typename T>
nnlib::Tensor<T> &operator/=(nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs);

template <typename T>
nnlib::Tensor<T> operator/(const nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template std::ostream &operator<<(std::ostream &, const nnlib::Tensor<NN_REAL_T> &);
	extern template nnlib::Tensor<NN_REAL_T> &operator+=(nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &);
	extern template nnlib::Tensor<NN_REAL_T> operator+(const nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &);
	extern template nnlib::Tensor<NN_REAL_T> &operator-=(nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &);
	extern template nnlib::Tensor<NN_REAL_T> operator-(const nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &);
	extern template nnlib::Tensor<NN_REAL_T> &operator*=(nnlib::Tensor<NN_REAL_T> &, NN_REAL_T);
	extern template nnlib::Tensor<NN_REAL_T> operator*(const nnlib::Tensor<NN_REAL_T> &, NN_REAL_T);
	extern template nnlib::Tensor<NN_REAL_T> &operator/=(nnlib::Tensor<NN_REAL_T> &, NN_REAL_T);
	extern template nnlib::Tensor<NN_REAL_T> operator/(const nnlib::Tensor<NN_REAL_T> &, NN_REAL_T);
#elif !defined NN_IMPL
	#include "tensor_operators.tpp"
#endif

#endif
