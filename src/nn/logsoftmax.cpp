#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/logsoftmax.hpp"
#include "nnlib/nn/detail/logsoftmax.tpp"

template class nnlib::LogSoftMax<NN_REAL_T>;

#endif
