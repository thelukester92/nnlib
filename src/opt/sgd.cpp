#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/opt/sgd.hpp"
#include "nnlib/opt/detail/sgd.tpp"

template class nnlib::SGD<NN_REAL_T>;

#endif
