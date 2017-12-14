#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/nn/module.hpp"
#include "nnlib/nn/detail/module.tpp"

template class nnlib::Module<NN_REAL_T>;

#endif
