#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/relu.hpp"
#include "nnlib/nn/detail/relu.tpp"

template class nnlib::ReLU<NN_REAL_T>;

#endif
