#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/concat.hpp"
#include "nnlib/nn/detail/concat.tpp"

template class nnlib::Concat<NN_REAL_T>;

#endif
