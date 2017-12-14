#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/linear.hpp"
#include "nnlib/nn/detail/linear.tpp"

template class nnlib::Linear<NN_REAL_T>;

#endif
