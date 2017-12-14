#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/lstm.hpp"
#include "nnlib/nn/detail/lstm.tpp"

template class nnlib::LSTM<NN_REAL_T>;

#endif
