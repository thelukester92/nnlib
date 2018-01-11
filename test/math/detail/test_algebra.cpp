#include "nnlib/core/error.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/math/algebra.hpp"
#include <algorithm>
#include <math.h>
#include <string>
#include <vector>
using namespace nnlib;

template <bool TransA, bool TransB, typename T>
void addMatrixMultiply(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha = 1, T beta = 1)
{
	for(size_t i = 0; i < M; ++i)
	{
		for(size_t j = 0; j < N; ++j)
		{
			C[i * ldc + j] *= beta;
			for(size_t k = 0; k < K; ++k)
			{
				C[i * ldc + j] += alpha * (TransA ? A[k * lda + i] : A[i * lda + k]) * (TransB ? B[j * ldb + k] : B[k * ldb + j]);
			}
		}
	}
}

template <typename T>
bool almostEqual(T a, T b)
{
	return fabs(a - b) < 1e-9;
}

template <>
bool almostEqual<float>(float a, float b)
{
	return fabs(a - b) < 1e-2;
}

template <typename T>
void TestAlgebraImpl(std::string name)
{
	Tensor<T> x(10), y(10);
	Tensor<T> A(10, 10), B(10, 10), C(10, 10), D(10, 10);

	Tensor<T> z = y.view(5, 2).select(1, 0);
	Tensor<T> E = B.view(5, 10, 2).select(2, 0);

	std::fill(x.begin(), x.end(), 0.5);
	Algebra<T>::vFill(y, 0.5);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), name + "::vFill with stride = 1 failed!");


	for(size_t i = 0; i < x.size(); i += 2)
		x(i) = 3.14;
	Algebra<T>::vFill(z, 3.14);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), name + "::vFill with stride = 2 failed!");

	for(size_t i = 0; i < x.size(); ++i)
		x(i) *= 2;
	Algebra<T>::vScale(y, 2);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), name + "::vScale with stride = 1 failed!");

	for(size_t i = 0; i < x.size(); i += 2)
		x(i) *= 0.75;
	Algebra<T>::vScale(z, 0.75);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), name + "::vScale with stride = 2 failed!");

	std::fill(A.begin(), A.end(), 0.5);
	Algebra<T>::mFill(B, 0.5);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), name + "::mFill with ld == cols failed!");

	for(size_t i = 0; i < 10; i += 2)
		for(size_t j = 0; j < 10; ++j)
			A(i, j) = 3.14;
	Algebra<T>::mFill(E, 3.14);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), name + "::mFill with ld != cols failed!");

	for(size_t i = 0; i < A.size(0); ++i)
		for(size_t j = 0; j < A.size(1); ++j)
			A(i, j) *= 2;
	Algebra<T>::mScale(B, 2);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), name + "::mScale with ld == cols failed!");

	for(size_t i = 0; i < 10; i += 2)
		for(size_t j = 0; j < 10; ++j)
			A(i, j) *= 0.75;
	Algebra<T>::mScale(E, 0.75);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), name + "::mScale with ld != cols failed!");

	std::fill(y.begin(), y.end(), 0.12345);
	Algebra<T>::vAdd_v(x, y);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual<T>(x(i) + 0.12345, y(i)), name + "::vAdd_v with stride = 1 failed!");

	Algebra<T>::vAdd_v(y, y);
	for(size_t i = 0; i < x.size(); ++i)
	{
		if(i % 2 == 0)
		{
			NNHardAssert(almostEqual<T>(x(i) + 0.12345 + x(i) + 0.12345, y(i)), name + "::vAdd_v with stride = 2 failed!");
		}
		else
		{
			NNHardAssert(almostEqual<T>(x(i) + 0.12345, y(i)), name + "::vAdd_v with stride = 2 failed!");
		}
	}

	std::fill(y.begin(), y.end(), 0.12345);
	Algebra<T>::vAdd_v(x, y, 0.5);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual<T>(0.5 * x(i) + 0.12345, y(i)), name + "::vAdd_v with alpha != 1 failed!");

	std::fill(y.begin(), y.end(), 0.12345);
	Algebra<T>::vAdd_v(x, y, 1, 0.75);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual<T>(x(i) + 0.75 * 0.12345, y(i)), name + "::vAdd_v with beta != 1 failed!");

	std::fill(A.begin(), A.end(), 0);
	Algebra<T>::mAdd_vv(x, y, A);
	for(size_t i = 0; i < x.size(); ++i)
		for(size_t j = 0; j < y.size(); ++j)
			NNHardAssert(almostEqual<T>(A(i, j), x(i) * y(j)), name + "::mAdd_vv with stride = 1 failed!");

	Algebra<T>::mAdd_vv(z, z, A);
	for(size_t i = 0; i < y.size(); ++i)
	{
		for(size_t j = 0; j < y.size(); ++j)
		{
			if(i < y.size() / 2 && j < y.size() / 2)
			{
				NNHardAssert(almostEqual<T>(A(i, j), y(i) * y(j) + y(2*i) * y(2*j)), name + "::mAdd_vv with stride = 2 failed!");
			}
			else
			{
				NNHardAssert(almostEqual<T>(A(i, j), y(i) * y(j)), name + "::mAdd_vv with stride = 2 failed!");
			}
		}
	}

	Algebra<T>::mAdd_vv(x, y, A, 1, 0);
	for(size_t i = 0; i < x.size(); ++i)
		for(size_t j = 0; j < y.size(); ++j)
			NNHardAssert(almostEqual<T>(A(i, j), x(i) * y(j)), name + "::mAdd_vv with beta != 1 failed!");

	std::fill(y.begin(), y.end(), 0);
	Algebra<T>::vAdd_mv(A, x, y);
	for(size_t i = 0; i < 10; ++i)
	{
		T value = 0.0;
		for(size_t j = 0; j < 10; ++j)
			value += A(i, j) * x(j);
		NNHardAssert(almostEqual<T>(y(i), value), name + "::vAdd_mv failed!");
	}

	std::fill(y.begin(), y.end(), 0);
	Algebra<T>::vAdd_mtv(A, x, y);
	for(size_t i = 0; i < 10; ++i)
	{
		T value = 0.0;
		for(size_t j = 0; j < 10; ++j)
			value += A(j, i) * x(j);
		NNHardAssert(almostEqual<T>(y(i), value), name + "::vAdd_mtv failed!");
	}

	std::fill(B.begin(), B.end(), 10);
	Algebra<T>::mAdd_m(A, B);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(B(i, j), A(i, j) + 10), name + "::mAdd_m failed!");

	std::fill(B.begin(), B.end(), 10);
	Algebra<T>::mAdd_mt(A, B);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(B(i, j), A(j, i) + 10), name + "::mAdd_mt failed!");

	std::fill(C.begin(), C.end(), 0);
	addMatrixMultiply<false, false, T>(10, 10, 10, A.ptr(), 10, B.ptr(), 10, C.ptr(), 10, 1, 0);
	Algebra<T>::mAdd_mm(A, B, D, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(C(i, j), D(i, j)), name + "::mAdd_mm failed!");

	addMatrixMultiply<true, false, T>(10, 10, 10, A.ptr(), 10, B.ptr(), 10, C.ptr(), 10, 1, 0);
	Algebra<T>::mAdd_mtm(A, B, D, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(C(i, j), D(i, j)), name + "::mAdd_mm failed!");

	addMatrixMultiply<false, true, T>(10, 10, 10, A.ptr(), 10, B.ptr(), 10, C.ptr(), 10, 1, 0);
	Algebra<T>::mAdd_mmt(A, B, D, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(C(i, j), D(i, j)), name + "::mAdd_mm failed!");

	addMatrixMultiply<true, true, T>(10, 10, 10, A.ptr(), 10, B.ptr(), 10, C.ptr(), 10, 1, 0);
	Algebra<T>::mAdd_mm(A, B, D, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(C(i, j), D(j, i)), name + "::mAdd_mm with both matrices transposed failed!");
}

void TestAlgebra()
{
#if defined NN_REAL_T && !defined NN_IMPL
#define NN_STR(s) #s
	TestAlgebraImpl<NN_REAL_T>(std::string("Algebra<") + NN_STR(NN_REAL_T) + ">");
#undef NN_STR
#elif !defined NN_IMPL
	TestAlgebraImpl<double>("Algebra<double>");
	TestAlgebraImpl<float>("Algebra<float>");
#endif
}
