#ifndef NN_ACCEL_CPU
	#warning "You are not using any CPU acceleration! Define NN_ACCEL to use BLAS."
#else
#ifdef NN_REAL_T
#ifndef MATH_ALGEBRA_BLAS_TPP
#define MATH_ALGEBRA_BLAS_TPP

#include "../algebra.hpp"
#include "nnlib/core/detail/tensor.tpp"

#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif

namespace nnlib
{

template <>
void Algebra<float>::vScale(Tensor<float> &_x, float alpha)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	float *x = _x.ptr();
	size_t n = _x.size(), s = _x.stride(0);
	cblas_sscal(n, alpha, x, s);
}

template <>
void Algebra<double>::vScale(Tensor<double> &_x, double alpha)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	double *x = _x.ptr();
	size_t n = _x.size(), s = _x.stride(0);
	cblas_dscal(n, alpha, x, s);
}

template <>
void Algebra<float>::mScale(Tensor<float> &_A, float alpha)
{
	NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
	float *A = _A.ptr();
	size_t r = _A.size(0), c = _A.size(1), ld = _A.stride(0);
	for(size_t i = 0; i < r; ++i, A += ld)
		cblas_sscal(c, alpha, A, 1);
}

template <>
void Algebra<double>::mScale(Tensor<double> &_A, double alpha)
{
	NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
	double *A = _A.ptr();
	size_t r = _A.size(0), c = _A.size(1), ld = _A.stride(0);
	for(size_t i = 0; i < r; ++i, A += ld)
		cblas_dscal(c, alpha, A, 1);
}

template <>
void Algebra<float>::vAdd_v(const Tensor<float> &_x, Tensor<float> &_y, float alpha)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
	const float *x = _x.ptr();
	float *y = _y.ptr();
	size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
	cblas_saxpy(n, alpha, x, sx, y, sy);
}

template <>
void Algebra<double>::vAdd_v(const Tensor<double> &_x, Tensor<double> &_y, double alpha)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
	const double *x = _x.ptr();
	double *y = _y.ptr();
	size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
	cblas_daxpy(n, alpha, x, sx, y, sy);
}

template <>
void Algebra<float>::vAdd_v(const Tensor<float> &_x, Tensor<float> &_y, float alpha, float beta)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
	const float *x = _x.ptr();
	float *y = _y.ptr();
	size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
	#ifdef __APPLE__
		catlas_saxpby(n, alpha, x, sx, beta, y, sy);
	#else
		cblas_saxpby(n, alpha, x, sx, beta, y, sy);
	#endif
}

template <>
void Algebra<double>::vAdd_v(const Tensor<double> &_x, Tensor<double> &_y, double alpha, double beta)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
	const double *x = _x.ptr();
	double *y = _y.ptr();
	size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
	#ifdef __APPLE__
		catlas_daxpby(n, alpha, x, sx, beta, y, sy);
	#else
		cblas_daxpby(n, alpha, x, sx, beta, y, sy);
	#endif
}

template <>
void Algebra<float>::mAdd_vv(const Tensor<float> &_x, const Tensor<float> &_y, Tensor<float> &_A, float alpha, float beta)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_y.dims(), 1, "Expected a vector!");
	NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
	NNAssertEquals(_x.size(), _A.size(0), "Incompatible operands!");
	NNAssertEquals(_y.size(), _A.size(1), "Incompatible operands!");
	const float *x = _x.ptr();
	const float *y = _y.ptr();
	float *A = _A.ptr();
	size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
	if(beta != 1)
		mScale(_A, beta);
	cblas_sger(CblasRowMajor, r, c, alpha, x, sx, y, sy, A, lda);
}

template <>
void Algebra<double>::mAdd_vv(const Tensor<double> &_x, const Tensor<double> &_y, Tensor<double> &_A, double alpha, double beta)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_y.dims(), 1, "Expected a vector!");
	NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
	NNAssertEquals(_x.size(), _A.size(0), "Incompatible operands!");
	NNAssertEquals(_y.size(), _A.size(1), "Incompatible operands!");
	const double *x = _x.ptr();
	const double *y = _y.ptr();
	double *A = _A.ptr();
	size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
	if(beta != 1)
		mScale(_A, beta);
	cblas_dger(CblasRowMajor, r, c, alpha, x, sx, y, sy, A, lda);
}

template <>
void Algebra<float>::vAdd_mv(const Tensor<float> &_A, const Tensor<float> &_x, Tensor<float> &_y, float alpha, float beta)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_y.dims(), 1, "Expected a vector!");
	NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
	NNAssertEquals(_x.size(), _A.size(1), "Incompatible operands!");
	NNAssertEquals(_y.size(), _A.size(0), "Incompatible operands!");
	const float *A = _A.ptr();
	const float *x = _x.ptr();
	float *y = _y.ptr();
	size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
	cblas_sgemv(CblasRowMajor, CblasNoTrans, r, c, alpha, A, lda, x, sx, beta, y, sy);
}

template <>
void Algebra<double>::vAdd_mv(const Tensor<double> &_A, const Tensor<double> &_x, Tensor<double> &_y, double alpha, double beta)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_y.dims(), 1, "Expected a vector!");
	NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
	NNAssertEquals(_x.size(), _A.size(1), "Incompatible operands!");
	NNAssertEquals(_y.size(), _A.size(0), "Incompatible operands!");
	const double *A = _A.ptr();
	const double *x = _x.ptr();
	double *y = _y.ptr();
	size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, r, c, alpha, A, lda, x, sx, beta, y, sy);
}

template <>
void Algebra<float>::vAdd_mtv(const Tensor<float> &_A, const Tensor<float> &_x, Tensor<float> &_y, float alpha, float beta)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_y.dims(), 1, "Expected a vector!");
	NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
	NNAssertEquals(_x.size(), _A.size(0), "Incompatible operands!");
	NNAssertEquals(_y.size(), _A.size(1), "Incompatible operands!");
	const float *A = _A.ptr();
	const float *x = _x.ptr();
	float *y = _y.ptr();
	size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
	cblas_sgemv(CblasRowMajor, CblasTrans, r, c, alpha, A, lda, x, sx, beta, y, sy);
}

template <>
void Algebra<double>::vAdd_mtv(const Tensor<double> &_A, const Tensor<double> &_x, Tensor<double> &_y, double alpha, double beta)
{
	NNAssertEquals(_x.dims(), 1, "Expected a vector!");
	NNAssertEquals(_y.dims(), 1, "Expected a vector!");
	NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
	NNAssertEquals(_x.size(), _A.size(0), "Incompatible operands!");
	NNAssertEquals(_y.size(), _A.size(1), "Incompatible operands!");
	const double *A = _A.ptr();
	const double *x = _x.ptr();
	double *y = _y.ptr();
	size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
	cblas_dgemv(CblasRowMajor, CblasTrans, r, c, alpha, A, lda, x, sx, beta, y, sy);
}

#ifndef NN_ACCEL_GPU
	template <>
	void Algebra<float>::mAdd_mm(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
	{
		NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_A.size(0), _C.size(0), "Incompatible operands!");
		NNAssertEquals(_A.size(1), _B.size(0), "Incompatible operands!");
		NNAssertEquals(_B.size(1), _C.size(1), "Incompatible operands!");
		float *A = const_cast<float *>(_A.ptr());
		float *B = const_cast<float *>(_B.ptr());
		float *C = _C.ptr();
		size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
		size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	template <>
	void Algebra<double>::mAdd_mm(const Tensor<double> &_A, const Tensor<double> &_B, Tensor<double> &_C, double alpha, double beta)
	{
		NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_A.size(0), _C.size(0), "Incompatible operands!");
		NNAssertEquals(_A.size(1), _B.size(0), "Incompatible operands!");
		NNAssertEquals(_B.size(1), _C.size(1), "Incompatible operands!");
		double *A = const_cast<double *>(_A.ptr());
		double *B = const_cast<double *>(_B.ptr());
		double *C = _C.ptr();
		size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
		size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	template <>
	void Algebra<float>::mAdd_mtm(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
	{
		NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_A.size(1), _C.size(0), "Incompatible operands!");
		NNAssertEquals(_A.size(0), _B.size(0), "Incompatible operands!");
		NNAssertEquals(_B.size(1), _C.size(1), "Incompatible operands!");
		float *A = const_cast<float *>(_A.ptr());
		float *B = const_cast<float *>(_B.ptr());
		float *C = _C.ptr();
		size_t M = _C.size(0), N = _C.size(1), K = _A.size(0);
		size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	template <>
	void Algebra<double>::mAdd_mtm(const Tensor<double> &_A, const Tensor<double> &_B, Tensor<double> &_C, double alpha, double beta)
	{
		NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_A.size(1), _C.size(0), "Incompatible operands!");
		NNAssertEquals(_A.size(0), _B.size(0), "Incompatible operands!");
		NNAssertEquals(_B.size(1), _C.size(1), "Incompatible operands!");
		double *A = const_cast<double *>(_A.ptr());
		double *B = const_cast<double *>(_B.ptr());
		double *C = _C.ptr();
		size_t M = _C.size(0), N = _C.size(1), K = _A.size(0);
		size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	template <>
	void Algebra<float>::mAdd_mmt(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
	{
		NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_A.size(0), _C.size(0), "Incompatible operands!");
		NNAssertEquals(_A.size(1), _B.size(1), "Incompatible operands!");
		NNAssertEquals(_B.size(0), _C.size(1), "Incompatible operands!");
		float *A = const_cast<float *>(_A.ptr());
		float *B = const_cast<float *>(_B.ptr());
		float *C = _C.ptr();
		size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
		size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	template <>
	void Algebra<double>::mAdd_mmt(const Tensor<double> &_A, const Tensor<double> &_B, Tensor<double> &_C, double alpha, double beta)
	{
		NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
		NNAssertEquals(_A.size(0), _C.size(0), "Incompatible operands!");
		NNAssertEquals(_A.size(1), _B.size(1), "Incompatible operands!");
		NNAssertEquals(_B.size(0), _C.size(1), "Incompatible operands!");
		double *A = const_cast<double *>(_A.ptr());
		double *B = const_cast<double *>(_B.ptr());
		double *C = _C.ptr();
		size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
		size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
#endif

}

#endif
#endif
#endif
