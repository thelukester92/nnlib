#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/dropconnect.hpp"
#include "nnlib/nn/detail/dropconnect.tpp"

template class nnlib::DropConnect<NN_REAL_T>;

#endif
