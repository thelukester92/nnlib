#ifndef MATH_TPP
#define MATH_TPP

#include "../math.hpp"

namespace nnlib
{

template <typename T>
void Math<T>::vFill(T *x, size_t n, size_t s, T alpha)
{
	for(size_t i = 0; i < n; ++i)
		x[i * s] = alpha;
}

template <typename T>
void Math<T>::vScale(T *x, size_t n, size_t s, T alpha)
{
	for(size_t i = 0; i < n; ++i)
		x[i * s] *= alpha;
}

template <typename T>
void Math<T>::mFill(T *A, size_t r, size_t c, size_t ld, T alpha)
{
	for(size_t i = 0; i < r; ++i)
		for(size_t j = 0; j < c; ++j)
			A[i * ld + j] = alpha;
}

template <typename T>
void Math<T>::mScale(T *A, size_t r, size_t c, size_t ld, T alpha)
{
	for(size_t i = 0; i < r; ++i)
		for(size_t j = 0; j < c; ++j)
			A[i * ld + j] *= alpha;
}

template <typename T>
void Math<T>::vAdd_v(const T *x, size_t n, size_t sx, T *y, size_t sy, T alpha)
{
	for(size_t i = 0; i < n; ++i)
		y[i * sy] = alpha * x[i * sx] + y[i * sy];
}

template <typename T>
void Math<T>::vAdd_v(const T *x, size_t n, size_t sx, T *y, size_t sy, T alpha, T beta)
{
	for(size_t i = 0; i < n; ++i)
		y[i * sy] = alpha * x[i * sx] + beta * y[i * sy];
}

template <typename T>
void Math<T>::mAdd_vv(const T *x, size_t r, size_t sx, const T *y, size_t c, size_t sy, T *A, size_t lda, T alpha, T beta)
{
	for(size_t i = 0; i < r; ++i)
		for(size_t j = 0; j < c; ++j)
			A[i * lda + j] = alpha * x[i * sx] * y[j * sy] + beta * A[i * lda + j];
}

template <typename T>
void Math<T>::vAdd_mv(const T *A, size_t ra, size_t ca, size_t lda, const T *x, size_t sx, T *y, size_t sy, T alpha, T beta)
{
	for(size_t i = 0; i < ra; ++i)
	{
		T &v = y[i * sy];
		v *= beta;
		for(size_t j = 0; j < ca; ++j)
			v += alpha * A[i * lda + j] * x[j * sx];
	}
}

template <typename T>
void Math<T>::vAdd_mtv(const T *A, size_t ra, size_t ca, size_t lda, const T *x, size_t sx, T *y, size_t sy, T alpha, T beta)
{
	for(size_t i = 0; i < ca; ++i)
	{
		T &v = y[i * sy];
		v *= beta;
		for(size_t j = 0; j < ra; ++j)
			v += alpha * A[j * lda + i] * x[j * sx];
	}
}

template <typename T>
void Math<T>::mAdd_m(const T *A, size_t r, size_t c, size_t lda, T *B, size_t ldb, T alpha, T beta)
{
	for(size_t i = 0; i < r; ++i)
		for(size_t j = 0; j < c; ++j)
			B[i * ldb + j] = alpha * A[i * lda + j] + beta * B[i * ldb + j];
}

template <typename T>
void Math<T>::mAdd_mt(const T *A, size_t r, size_t c, size_t lda, T *B, size_t ldb, T alpha, T beta)
{
	for(size_t i = 0; i < r; ++i)
		for(size_t j = 0; j < c; ++j)
			B[i * ldb + j] = alpha * A[j * lda + i] + beta * B[i * ldb + j];
}

template <typename T>
void Math<T>::mAdd_mm(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha, T beta)
{
	for(size_t i = 0; i < M; ++i)
	{
		for(size_t j = 0; j < N; ++j)
		{
			T &v = C[i * ldc + j];
			v *= beta;
			for(size_t k = 0; k < K; ++k)
			{
				v += alpha * A[i * lda + k] * B[k * ldb + j];
			}
		}
	}
}

template <typename T>
void Math<T>::mAdd_mtm(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha, T beta)
{
	for(size_t i = 0; i < M; ++i)
	{
		for(size_t j = 0; j < N; ++j)
		{
			T &v = C[i * ldc + j];
			v *= beta;
			for(size_t k = 0; k < K; ++k)
			{
				v += alpha * A[k * lda + i] * B[k * ldb + j];
			}
		}
	}
}

template <typename T>
void Math<T>::mAdd_mmt(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha, T beta)
{
	for(size_t i = 0; i < M; ++i)
	{
		for(size_t j = 0; j < N; ++j)
		{
			T &v = C[i * ldc + j];
			v *= beta;
			for(size_t k = 0; k < K; ++k)
			{
				v += alpha * A[i * lda + k] * B[j * ldb + k];
			}
		}
	}
}

}

#endif
