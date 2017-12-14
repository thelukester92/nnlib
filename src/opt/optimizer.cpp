#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/opt/optimizer.hpp"
#include "nnlib/opt/detail/optimizer.tpp"

template class nnlib::Optimizer<NN_REAL_T>;

#endif
