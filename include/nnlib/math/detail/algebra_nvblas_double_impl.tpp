#ifndef MATH_ALGEBRA_NVBLAS_DOUBLE_IMPL_TPP
#define MATH_ALGEBRA_NVBLAS_DOUBLE_IMPL_TPP

#include "../algebra.hpp"
#include "nnlib/core/detail/tensor.tpp"

#include <nvblas.h>

#define gemm dgemm
#include "algebra_nvblas_impl.tpp"
#undef gemm

#endif
