#ifndef MATH_ALGEBRA_NVBLAS_DOUBLE_IMPL_TPP
#define MATH_ALGEBRA_NVBLAS_DOUBLE_IMPL_TPP

#include "../algebra.hpp"
#include "nnlib/core/detail/tensor.tpp"

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#define gemm dgemm
#include "algebra_blas_impl.tpp"
#undef gemm

#endif
