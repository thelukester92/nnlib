#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/batchnorm.hpp"
#include "nnlib/nn/detail/batchnorm.tpp"

template class nnlib::BatchNorm<NN_REAL_T>;

#endif
