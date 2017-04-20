#ifndef ALGEBRA_H
#define ALGEBRA_H

#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#endif

#include "error.h"
#include "tensor.h"

namespace nnlib
{

/// BLAS wrapper to move type out to a template parameter.
template <typename T>
class BLAS
{};

/// Double-precision BLAS.
template <>
class BLAS<double>
{
using T = double;
public:
	/// Matrix x Matrix
	static void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, size_t M, size_t N, size_t K, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_dgemm(o, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	
	/// Matrix x Vector
	static void gemv(CBLAS_ORDER o, CBLAS_TRANSPOSE trans, size_t M, size_t N, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_dgemv(o, trans, M, N, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
	
	/// Vector x Vector (outer product)
	static void ger(CBLAS_ORDER o, size_t M, size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY, T *A, size_t lda)
	{
		cblas_dger(o, M, N, alpha, X, strideX, Y, strideY, A, lda);
	}
	
	/// Vector + Vector (scaled)
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_daxpy(N, alpha, X, strideX, Y, strideY);
	}
};

/// Single-precision BLAS.
template <>
class BLAS<float>
{
using T = float;
public:
	/// Matrix x Matrix
	static void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, size_t M, size_t N, size_t K, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_sgemm(o, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	
	/// Matrix x Vector
	static void gemv(CBLAS_ORDER o, CBLAS_TRANSPOSE trans, size_t M, size_t N, T alpha, T *A, size_t lda, T *X, size_t strideX, T beta, T *Y, size_t strideY)
	{
		cblas_sgemv(o, trans, M, N, alpha, A, lda, X, strideX, beta, Y, strideY);
	}
	
	/// Vector x Vector (outer product)
	static void ger(CBLAS_ORDER o, size_t M, size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY, T *A, size_t lda)
	{
		cblas_sger(o, M, N, alpha, X, strideX, Y, strideY, A, lda);
	}
	
	/// Vector + Vector (scaled)
	static void axpy(size_t N, T alpha, T *X, size_t strideX, T *Y, size_t strideY)
	{
		cblas_saxpy(N, alpha, X, strideX, Y, strideY);
	}
};

/// Abstracted BLAS using Tensors with runtime checks.
template <typename T>
class Algebra
{
public:
	
	/// Matrix multiplication. Multiplies output matrix C by beta before adding AB.
	static void gemm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, bool tA = false, bool tB = false, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && C.dims() == 2, "Cannot call gemm on a non-matrix!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && C.stride(1) == 1, "Cannot call gemm on a matrix with non-contiguous columns!");
		
		size_t inner = A.size(1);
		CBLAS_TRANSPOSE transA = CblasNoTrans, transB = CblasNoTrans;
		
		if(tA)
		{
			inner = A.size(0);
			transA = CblasTrans;
		}
		
		if(tB)
		{
			transB = CblasTrans;
		}
		
		NNAssert((tA ? A.size(1) : A.size(0)) == C.size(0), "Incompatible rows!");
		NNAssert((tB ? B.size(0) : B.size(1)) == C.size(1), "Incompatible columns!");
		NNAssert((tB ? B.size(1) : B.size(0)) == inner, "Incompatible inner dimension!");
		
		BLAS<T>::gemm(
			CblasRowMajor, transA, transB, C.size(0), C.size(1), inner, alpha,
			const_cast<T *>(A.ptr()), A.size(1),
			const_cast<T *>(B.ptr()), B.size(1),
			beta, C.ptr(), C.size(1)
		);
	}
	
	/// Matrix vector multiplication. Multiplies output vector C by beta before adding AB.
	static void gemv(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, bool trans = false, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 1 && C.dims() == 1, "Incompatible dimensions for gemv!");
		
		size_t primaryDim = A.size(1);
		CBLAS_TRANSPOSE transA = CblasNoTrans;
		
		if(trans)
		{
			primaryDim = A.size(0);
			transA = CblasTrans;
		}
		
		BLAS<T>::gemv(
			CblasRowMajor, transA, A.size(0), A.size(1), alpha,
			const_cast<T *>(A.ptr()), A.size(1),
			const_cast<T *>(B.ptr()), B.stride(0),
			beta, C.ptr(), C.stride(0)
		);
	}
	
	/// Vector outer product. Adds AB to C; need to reset C manually if necessary.
	static void ger(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
	{
		NNAssert(A.dims() == 1 && B.dims() == 1, "Cannot call ger with non-vector operands!");
		NNAssert(C.dims() == 2, "Cannot call ger with a non-matrix resultant!");
		NNAssert(C.stride(1) == 1, "Cannot call ger on a matrix with non-contiguous columns!");
		NNAssert(A.size() == C.size(0), "Incompatible rows!");
		NNAssert(B.size() == C.size(1), "Incompatible columns!");
		
		BLAS<T>::ger(
			CblasRowMajor, A.size(), B.size(), 1,
			const_cast<T *>(A.ptr()), A.stride(0),
			const_cast<T *>(B.ptr()), B.stride(0),
			C.ptr(), C.stride(0)
		);
	}
	
	/// Scaled vector addition. Need to reset B manually if necessary.
	static void axpy(const Tensor<T> &A, Tensor<T> &B, T alpha = 1)
	{
		NNAssert(A.dims() == 1 && B.dims() == 1, "Cannot call axpy with non-vector operands!");
		NNAssert(A.size() == B.size(), "Incompatible operands for axpy!");
		BLAS<T>::axpy(A.size(), alpha, const_cast<T *>(A.ptr()), A.stride(0), B.ptr(), B.stride(0));
	}
};

}

#endif
