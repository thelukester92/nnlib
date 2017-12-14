#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/softmax.hpp"
#include "nnlib/nn/detail/softmax.tpp"

template class nnlib::SoftMax<NN_REAL_T>;

#endif
