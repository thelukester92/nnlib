#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/math/math.hpp"
#include "nnlib/math/detail/math.tpp"
#include "nnlib/math/detail/blas.tpp"
#include "nnlib/math/detail/nvblas.tpp"

template class nnlib::Math<NN_REAL_T>;

#endif
