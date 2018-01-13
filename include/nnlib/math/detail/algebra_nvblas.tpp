#ifdef NN_ACCEL_GPU
#ifndef MATH_ALGEBRA_NVBLAS_TPP
#define MATH_ALGEBRA_NVBLAS_TPP

#include "../algebra.hpp"
#include <nvblas.h>

namespace nnlib
{

template <>
void Algebra<float>::mAdd_mm(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    sgemm("N", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Algebra<double>::mAdd_mm(const Tensor<double> &_A, const Tensor<double> &_B, Tensor<double> &_C, double alpha, double beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    dgemm("N", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Algebra<float>::mAdd_mtm(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    sgemm("N", "T", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Algebra<double>::mAdd_mtm(const Tensor<double> &_A, const Tensor<double> &_B, Tensor<double> &_C, double alpha, double beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    dgemm("N", "T", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Algebra<float>::mAdd_mmt(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    sgemm("T", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Algebra<double>::mAdd_mmt(const Tensor<double> &_A, const Tensor<double> &_B, Tensor<double> &_C, double alpha, double beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    dgemm("T", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

}

#endif
#endif
