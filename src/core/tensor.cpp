#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/core/tensor.hpp"
#include "nnlib/core/detail/tensor.tpp"

template class nnlib::Tensor<NN_REAL_T>;

#endif
