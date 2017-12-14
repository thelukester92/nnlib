#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/sequential.hpp"
#include "nnlib/nn/detail/sequential.tpp"

template class nnlib::Sequential<NN_REAL_T>;

#endif
