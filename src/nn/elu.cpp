#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/elu.hpp"
#include "nnlib/nn/detail/elu.tpp"

template class nnlib::ELU<NN_REAL_T>;

#endif
