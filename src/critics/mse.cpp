#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/critics/mse.hpp"
#include "nnlib/critics/detail/mse.tpp"

template class nnlib::MSE<NN_REAL_T>;

#endif
