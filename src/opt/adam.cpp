#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/opt/adam.hpp"
#include "nnlib/opt/detail/adam.tpp"

template class nnlib::Adam<NN_REAL_T>;

#endif
