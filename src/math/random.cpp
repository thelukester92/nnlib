#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/math/random.hpp"
#include "nnlib/math/detail/random.tpp"

template class nnlib::Random<NN_REAL_T>;

#endif
