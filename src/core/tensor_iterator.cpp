#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/core/detail/tensor_iterator.hpp"
#include "nnlib/core/detail/tensor_iterator.tpp"

template class nnlib::TensorIterator<NN_REAL_T>;
template class nnlib::TensorIterator<const NN_REAL_T>;

#endif
