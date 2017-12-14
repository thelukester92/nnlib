#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/logistic.hpp"
#include "nnlib/nn/detail/logistic.tpp"

template class nnlib::Logistic<NN_REAL_T>;

#endif
