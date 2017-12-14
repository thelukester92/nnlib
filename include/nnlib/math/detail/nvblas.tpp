#ifdef NN_ACCEL_GPU
#ifndef NVBLAS_TPP
#define NVBLAS_TPP

#include <nvblas.h>

namespace nnlib
{

template <>
void Math<float>::mAdd_mm(size_t M, size_t N, size_t K, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc, float alpha, float beta)
{
	int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
	sgemm("N", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Math<double>::mAdd_mm(size_t M, size_t N, size_t K, const double *A, size_t lda, const double *B, size_t ldb, double *C, size_t ldc, double alpha, double beta)
{
	int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
	dgemm("N", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Math<float>::mAdd_mtm(size_t M, size_t N, size_t K, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc, float alpha, float beta)
{
	int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
	sgemm("N", "T", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Math<double>::mAdd_mtm(size_t M, size_t N, size_t K, const double *A, size_t lda, const double *B, size_t ldb, double *C, size_t ldc, double alpha, double beta)
{
	int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
	dgemm("N", "T", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Math<float>::mAdd_mmt(size_t M, size_t N, size_t K, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc, float alpha, float beta)
{
	int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
	sgemm("T", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <>
void Math<float>::mAdd_mmt(size_t M, size_t N, size_t K, const double *A, size_t lda, const double *B, size_t ldb, double *C, size_t ldc, double alpha, double beta)
{
	int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
	sgemm("T", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

}

#endif
#endif
