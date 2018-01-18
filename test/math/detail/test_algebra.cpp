#include "../test_algebra.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/core/detail/tensor.tpp"
#include "nnlib/math/algebra.hpp"
#include "nnlib/math/detail/algebra.tpp"
using namespace nnlib;

#define T NN_REAL_T
NNTestClassImpl(Algebra)
{
    #include "test_algebra.tpp"
}

#undef T
#define T float
NNTestClassImpl(Algebra_BLAS_float)
{
    #include "test_algebra.tpp"
}

#undef T
#define T double
NNTestClassImpl(Algebra_BLAS_double)
{
    #include "test_algebra.tpp"
}

#undef T
