#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/prelu.hpp"
#include "nnlib/nn/detail/prelu.tpp"

template class nnlib::PReLU<NN_REAL_T>;

#endif
