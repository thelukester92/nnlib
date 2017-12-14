#ifndef NN_ACCEL
	#warning "You are not using any acceleration! Define NN_ACCEL to use BLAS."
#else
#ifdef NN_REAL_T
#ifndef BLAS_TPP
#define BLAS_TPP

#include "../math.hpp"

#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif

namespace nnlib
{

template <>
void Math<float>::vScale(float *x, size_t n, size_t s, float alpha)
{
	cblas_sscal(n, alpha, x, s);
}

template <>
void Math<double>::vScale(double *x, size_t n, size_t s, double alpha)
{
	cblas_dscal(n, alpha, x, s);
}

template <>
void Math<float>::mScale(float *A, size_t r, size_t c, size_t ld, float alpha)
{
	for(size_t i = 0; i < r; ++i, A += ld)
		cblas_sscal(c, alpha, A, 1);
}

template <>
void Math<double>::mScale(double *A, size_t r, size_t c, size_t ld, double alpha)
{
	for(size_t i = 0; i < r; ++i, A += ld)
		cblas_dscal(c, alpha, A, 1);
}

template <>
void Math<float>::vAdd_v(const float *x, size_t n, size_t sx, float *y, size_t sy, float alpha)
{
	cblas_saxpy(n, alpha, x, sx, y, sy);
}

template <>
void Math<double>::vAdd_v(const double *x, size_t n, size_t sx, double *y, size_t sy, double alpha)
{
	cblas_daxpy(n, alpha, x, sx, y, sy);
}

template <>
void Math<float>::vAdd_v(const float *x, size_t n, size_t sx, float *y, size_t sy, float alpha, float beta)
{
	#ifdef __APPLE__
		catlas_saxpby(n, alpha, x, sx, beta, y, sy);
	#else
		cblas_saxpby(n, alpha, x, sx, beta, y, sy);
	#endif
}

template <>
void Math<double>::vAdd_v(const double *x, size_t n, size_t sx, double *y, size_t sy, double alpha, double beta)
{
	#ifdef __APPLE__
		catlas_daxpby(n, alpha, x, sx, beta, y, sy);
	#else
		cblas_daxpby(n, alpha, x, sx, beta, y, sy);
	#endif
}

template <>
void Math<float>::mAdd_vv(const float *x, size_t r, size_t sx, const float *y, size_t c, size_t sy, float *A, size_t lda, float alpha, float beta)
{
	if(beta != 1)
		mScale(A, r, c, lda, beta);
	cblas_sger(CblasRowMajor, r, c, alpha, x, sx, y, sy, A, lda);
}

template <>
void Math<double>::mAdd_vv(const double *x, size_t r, size_t sx, const double *y, size_t c, size_t sy, double *A, size_t lda, double alpha, double beta)
{
	if(beta != 1)
		mScale(A, r, c, lda, beta);
	cblas_dger(CblasRowMajor, r, c, alpha, x, sx, y, sy, A, lda);
}

template <>
void Math<float>::vAdd_mv(const float *A, size_t ra, size_t ca, size_t lda, const float *x, size_t sx, float *y, size_t sy, float alpha, float beta)
{
	cblas_sgemv(CblasRowMajor, CblasNoTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
}

template <>
void Math<double>::vAdd_mv(const double *A, size_t ra, size_t ca, size_t lda, const double *x, size_t sx, double *y, size_t sy, double alpha, double beta)
{
	cblas_dgemv(CblasRowMajor, CblasNoTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
}

template <>
void Math<float>::vAdd_mtv(const float *A, size_t ra, size_t ca, size_t lda, const float *x, size_t sx, float *y, size_t sy, float alpha, float beta)
{
	cblas_sgemv(CblasRowMajor, CblasTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
}

template <>
void Math<double>::vAdd_mtv(const double *A, size_t ra, size_t ca, size_t lda, const double *x, size_t sx, double *y, size_t sy, double alpha, double beta)
{
	cblas_dgemv(CblasRowMajor, CblasTrans, ra, ca, alpha, A, lda, x, sx, beta, y, sy);
}

template <>
void Math<float>::mAdd_mm(size_t M, size_t N, size_t K, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc, float alpha, float beta)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, const_cast<float *>(A), lda, const_cast<float *>(B), ldb, beta, C, ldc);
}

template <>
void Math<double>::mAdd_mm(size_t M, size_t N, size_t K, const double *A, size_t lda, const double *B, size_t ldb, double *C, size_t ldc, double alpha, double beta)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, const_cast<double *>(A), lda, const_cast<double *>(B), ldb, beta, C, ldc);
}

template <>
void Math<float>::mAdd_mtm(size_t M, size_t N, size_t K, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc, float alpha, float beta)
{
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, const_cast<float *>(A), lda, const_cast<float *>(B), ldb, beta, C, ldc);
}

template <>
void Math<double>::mAdd_mtm(size_t M, size_t N, size_t K, const double *A, size_t lda, const double *B, size_t ldb, double *C, size_t ldc, double alpha, double beta)
{
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, const_cast<double *>(A), lda, const_cast<double *>(B), ldb, beta, C, ldc);
}

template <>
void Math<float>::mAdd_mmt(size_t M, size_t N, size_t K, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc, float alpha, float beta)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, const_cast<float *>(A), lda, const_cast<float *>(B), ldb, beta, C, ldc);
}

template <>
void Math<double>::mAdd_mmt(size_t M, size_t N, size_t K, const double *A, size_t lda, const double *B, size_t ldb, double *C, size_t ldc, double alpha, double beta)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, const_cast<double *>(A), lda, const_cast<double *>(B), ldb, beta, C, ldc);
}

}

#endif
#endif
#endif
