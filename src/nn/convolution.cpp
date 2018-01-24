#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/convolution.hpp"
#include "nnlib/nn/detail/convolution.tpp"

template class nnlib::Convolution<NN_REAL_T>;

#endif
