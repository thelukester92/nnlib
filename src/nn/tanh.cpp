#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/tanh.hpp"
#include "nnlib/nn/detail/tanh.tpp"

template class nnlib::TanH<NN_REAL_T>;

#endif
