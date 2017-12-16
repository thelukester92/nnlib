#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/core/tensor.hpp"
#include "nnlib/core/detail/tensor.tpp"
#include "nnlib/core/detail/tensor_iterator.tpp"
#include "nnlib/core/detail/tensor_operators.tpp"

template class nnlib::Tensor<NN_REAL_T>;

#endif
