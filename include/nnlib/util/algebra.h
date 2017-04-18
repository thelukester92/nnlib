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
	static void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, size_t M, size_t N, size_t K, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_dgemm(o, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
};

/// Single-precision BLAS.
template <>
class BLAS<float>
{
using T = float;
public:
	static void gemm(CBLAS_ORDER o, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, size_t M, size_t N, size_t K, T alpha, T *A, size_t lda, T *B, size_t ldb, T beta, T *C, size_t ldc)
	{
		cblas_sgemm(o, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
};

/// Abstracted BLAS using Tensors with runtime checks.
template <typename T>
class Algebra
{
public:
	template <bool TransA, bool TransB>
	static void gemm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && C.dims() == 2, "Cannot call gemm on a non-matrix!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && C.stride(1) == 1, "Cannot call gemm on a matrix with non-contiguous columns!");
		
		CBLAS_TRANSPOSE tA = CblasNoTrans, tB = CblasNoTrans;
		size_t inner = A.size(1);
		
		if(TransA)
		{
			NNAssert(A.size(1) == C.size(0), "Incompatible rows!");
			tA = CblasTrans;
			inner = A.size(0);
		}
		else
		{
			NNAssert(A.size(0) == C.size(0), "Incompatible rows!");
		}
		
		if(TransB)
		{
			NNAssert(B.size(0) == C.size(1), "Incompatible columns!");
			NNAssert(B.size(1) == inner, "Incompatible inner dimension!");
			tB = CblasTrans;
		}
		else
		{
			NNAssert(B.size(1) == C.size(1), "Incompatible columns!");
			NNAssert(B.size(0) == inner, "Incompatible inner dimension!");
		}
		
		BLAS<T>::gemm(
			CblasRowMajor, tA, tB, C.size(0), C.size(1), inner, 1,
			const_cast<T *>(A.storage().ptr()), A.size(1),
			const_cast<T *>(B.storage().ptr()), B.size(1),
			0, C.storage().ptr(), C.size(1)
		);
	}
};

}

#endif
