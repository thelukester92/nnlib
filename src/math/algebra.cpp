#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/math/algebra.hpp"
#include "nnlib/math/detail/algebra.tpp"
#include "nnlib/math/detail/algebra_blas.tpp"
#include "nnlib/math/detail/algebra_nvblas.tpp"

template class nnlib::Algebra<NN_REAL_T>;

#endif
