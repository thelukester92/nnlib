#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/dropout.hpp"
#include "nnlib/nn/detail/dropout.tpp"

template class nnlib::Dropout<NN_REAL_T>;

#endif
