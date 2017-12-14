#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/opt/nadam.hpp"
#include "nnlib/opt/detail/nadam.tpp"

template class nnlib::Nadam<NN_REAL_T>;

#endif
