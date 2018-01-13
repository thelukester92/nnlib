#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/core/detail/tensor_operators.hpp"
#include "nnlib/core/detail/tensor_operators.tpp"

template std::ostream &operator<<(std::ostream &, const nnlib::Tensor<NN_REAL_T> &);
template nnlib::Tensor<NN_REAL_T> &operator+=(nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &);
template nnlib::Tensor<NN_REAL_T> operator+(const nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &);
template nnlib::Tensor<NN_REAL_T> &operator-=(nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &);
template nnlib::Tensor<NN_REAL_T> operator-(const nnlib::Tensor<NN_REAL_T> &, const nnlib::Tensor<NN_REAL_T> &);
template nnlib::Tensor<NN_REAL_T> &operator*=(nnlib::Tensor<NN_REAL_T> &, NN_REAL_T);
template nnlib::Tensor<NN_REAL_T> operator*(const nnlib::Tensor<NN_REAL_T> &, NN_REAL_T);
template nnlib::Tensor<NN_REAL_T> &operator/=(nnlib::Tensor<NN_REAL_T> &, NN_REAL_T);
template nnlib::Tensor<NN_REAL_T> operator/(const nnlib::Tensor<NN_REAL_T> &, NN_REAL_T);

#endif
