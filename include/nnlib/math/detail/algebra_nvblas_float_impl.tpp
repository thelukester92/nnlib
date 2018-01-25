#ifndef MATH_ALGEBRA_NVBLAS_FLOAT_IMPL_TPP
#define MATH_ALGEBRA_NVBLAS_FLOAT_IMPL_TPP

#include "../algebra.hpp"
#include "nnlib/core/detail/tensor.tpp"

#include <nvblas.h>

#define gemm sgemm
#include "algebra_nvblas_impl.tpp"
#undef gemm

#endif
