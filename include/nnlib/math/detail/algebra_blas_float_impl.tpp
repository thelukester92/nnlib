#ifndef MATH_ALGEBRA_BLAS_FLOAT_IMPL_TPP
#define MATH_ALGEBRA_BLAS_FLOAT_IMPL_TPP

#include "../algebra.hpp"
#include "nnlib/core/detail/tensor.tpp"

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#define scal  cblas_sscal
#define axpy  cblas_saxpy
#define axpby cblas_saxpby
#define ger   cblas_sger
#define gemv  cblas_sgemv
#define gemm  cblas_sgemm

#ifdef __APPLE__
    #undef axpby
    #define axpby catlas_saxpby
#endif

#include "algebra_blas_impl.tpp"

#undef scal
#undef axpy
#undef axpby
#undef ger
#undef gemv
#undef gemm

#endif
