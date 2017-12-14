#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/sin.hpp"
#include "nnlib/nn/detail/sin.tpp"

template class nnlib::Sin<NN_REAL_T>;

#endif
