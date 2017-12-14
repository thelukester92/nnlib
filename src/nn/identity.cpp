#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/identity.hpp"
#include "nnlib/nn/detail/identity.tpp"

template class nnlib::Identity<NN_REAL_T>;

#endif
