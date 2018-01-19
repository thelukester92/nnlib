#ifndef MATH_ALGEBRA_BLAS_DOUBLE_IMPL_TPP
#define MATH_ALGEBRA_BLAS_DOUBLE_IMPL_TPP

#include "../algebra.hpp"
#include "nnlib/core/detail/tensor.tpp"

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#define scal  cblas_dscal
#define axpy  cblas_daxpy
#define axpby cblas_daxpby
#define ger   cblas_dger
#define gemv  cblas_dgemv
#define gemm  cblas_dgemm

#ifdef __APPLE__
    #undef axpby
    #define axpby catlas_daxpby
#endif

#include "algebra_blas_impl.tpp"

#undef scal
#undef axpy
#undef axpby
#undef ger
#undef gemv
#undef gemm

#endif
