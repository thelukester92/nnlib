#ifndef TEST_MATH
#define TEST_MATH

#include "nnlib/math/math.hpp"
#include <algorithm>
#include <vector>
#include <string>
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

template <NN_REAL_T>
bool almostEqual<float>(float a, float b)
{
	return fabs(a - b) < 1e-2;
}

template <typename T>
void TestMathImpl(std::string name)
{
	std::vector<T> x(10), y(10);
	std::vector<T> A(100), B(100), C(100), D(100);
	
	std::fill(x.begin(), x.end(), 0.5);
	Math<T>::vFill(y.data(), y.size(), 1, 0.5);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), name + "::vFill with stride = 1 failed!");
	
	for(size_t i = 0; i < x.size(); i += 2)
		x[i] = 3.14;
	Math<T>::vFill(y.data(), y.size() / 2, 2, 3.14);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), name + "::vFill with stride = 2 failed!");
	
	for(size_t i = 0; i < x.size(); ++i)
		x[i] *= 2;
	Math<T>::vScale(y.data(), y.size(), 1, 2);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), name + "::vScale with stride = 1 failed!");
	
	for(size_t i = 0; i < x.size(); i += 2)
		x[i] *= 0.75;
	Math<T>::vScale(y.data(), y.size() / 2, 2, 0.75);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), name + "::vScale with stride = 2 failed!");
	
	std::fill(A.begin(), A.end(), 0.5);
	Math<T>::mFill(B.data(), 10, 10, 10, 0.5);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), name + "::mFill with ld == cols failed!");
	
	for(size_t i = 0; i < 10; i += 2)
		for(size_t j = 0; j < 10; ++j)
			A[i * 10 + j] = 3.14;
	Math<T>::mFill(B.data(), 5, 10, 20, 3.14);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), name + "::mFill with ld != cols failed!");
	
	for(size_t i = 0; i < A.size(); ++i)
		A[i] *= 2;
	Math<T>::mScale(B.data(), 10, 10, 10, 2);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), name + "::mScale with ld == cols failed!");
	
	for(size_t i = 0; i < 10; i += 2)
		for(size_t j = 0; j < 10; ++j)
			A[i * 10 + j] *= 0.75;
	Math<T>::mScale(B.data(), 5, 10, 20, 0.75);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), name + "::mScale with ld != cols failed!");
	
	std::fill(y.begin(), y.end(), 0.12345);
	Math<T>::vAdd_v(x.data(), x.size(), 1, y.data(), 1);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual<T>(x[i] + 0.12345, y[i]), name + "::vAdd_v with stride = 1 failed!");
	
	Math<T>::vAdd_v(y.data(), y.size() / 2, 2, y.data(), 2);
	for(size_t i = 0; i < x.size(); ++i)
	{
		if(i % 2 == 0)
		{
			NNHardAssert(almostEqual<T>(x[i] + 0.12345 + x[i] + 0.12345, y[i]), name + "::vAdd_v with stride = 2 failed!");
		}
		else
		{
			NNHardAssert(almostEqual<T>(x[i] + 0.12345, y[i]), name + "::vAdd_v with stride = 2 failed!");
		}
	}
	
	std::fill(y.begin(), y.end(), 0.12345);
	Math<T>::vAdd_v(x.data(), x.size(), 1, y.data(), 1, 0.5);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual<T>(0.5 * x[i] + 0.12345, y[i]), name + "::vAdd_v with alpha != 1 failed!");
	
	std::fill(y.begin(), y.end(), 0.12345);
	Math<T>::vAdd_v(x.data(), x.size(), 1, y.data(), 1, 1, 0.75);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual<T>(x[i] + 0.75 * 0.12345, y[i]), name + "::vAdd_v with beta != 1 failed!");
	
	std::fill(A.begin(), A.end(), 0);
	Math<T>::mAdd_vv(x.data(), x.size(), 1, y.data(), y.size(), 1, A.data(), 10);
	for(size_t i = 0; i < x.size(); ++i)
		for(size_t j = 0; j < y.size(); ++j)
			NNHardAssert(almostEqual<T>(A[i * 10 + j], x[i] * y[j]), name + "::mAdd_vv with stride = 1 failed!");
	
	Math<T>::mAdd_vv(x.data(), x.size() / 2, 2, y.data(), y.size() / 2, 2, A.data(), 10);
	for(size_t i = 0; i < x.size(); ++i)
	{
		for(size_t j = 0; j < y.size(); ++j)
		{
			if(i < x.size() / 2 && j < y.size() / 2)
			{
				NNHardAssert(almostEqual<T>(A[i * 10 + j], x[i] * y[j] + x[2*i] * y[2*j]), name + "::mAdd_vv with stride = 2 failed!");
			}
			else
			{
				NNHardAssert(almostEqual<T>(A[i * 10 + j], x[i] * y[j]), name + "::mAdd_vv with stride = 2 failed!");
			}
		}
	}
	
	Math<T>::mAdd_vv(x.data(), x.size(), 1, y.data(), y.size(), 1, A.data(), 10, 1, 0);
	for(size_t i = 0; i < x.size(); ++i)
		for(size_t j = 0; j < y.size(); ++j)
			NNHardAssert(almostEqual<T>(A[i * 10 + j], x[i] * y[j]), name + "::mAdd_vv with beta != 1 failed!");
	
	std::fill(y.begin(), y.end(), 0);
	Math<T>::vAdd_mv(A.data(), 10, 10, 10, x.data(), 1, y.data(), 1);
	for(size_t i = 0; i < 10; ++i)
	{
		T value = 0.0;
		for(size_t j = 0; j < 10; ++j)
		{
			value += A[i * 10 + j] * x[j];
		}
		NNHardAssert(almostEqual<T>(y[i], value), name + "::vAdd_mv failed!");
	}
	
	std::fill(y.begin(), y.end(), 0);
	Math<T>::vAdd_mtv(A.data(), 10, 10, 10, x.data(), 1, y.data(), 1);
	for(size_t i = 0; i < 10; ++i)
	{
		T value = 0.0;
		for(size_t j = 0; j < 10; ++j)
		{
			value += A[j * 10 + i] * x[j];
		}
		NNHardAssert(almostEqual<T>(y[i], value), name + "::vAdd_mtv failed!");
	}
	
	std::fill(B.begin(), B.end(), 10);
	Math<T>::mAdd_m(A.data(), 10, 10, 10, B.data(), 10);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(B[i * 10 + j], A[i * 10 + j] + 10), name + "::mAdd_m failed!");
	
	std::fill(B.begin(), B.end(), 10);
	Math<T>::mAdd_mt(A.data(), 10, 10, 10, B.data(), 10);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(B[i * 10 + j], A[j * 10 + i] + 10), name + "::mAdd_mt failed!");
	
	std::fill(C.begin(), C.end(), 0);
	addMatrixMultiply<false, false, T>(10, 10, 10, A.data(), 10, B.data(), 10, C.data(), 10, 1, 0);
	Math<T>::mAdd_mm(10, 10, 10, A.data(), 10, B.data(), 10, D.data(), 10, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(C[i * 10 + j], D[i * 10 + j]), name + "::mAdd_mm failed!");
	
	addMatrixMultiply<true, false, T>(10, 10, 10, A.data(), 10, B.data(), 10, C.data(), 10, 1, 0);
	Math<T>::mAdd_mtm(10, 10, 10, A.data(), 10, B.data(), 10, D.data(), 10, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(C[i * 10 + j], D[i * 10 + j]), name + "::mAdd_mm failed!");
	
	addMatrixMultiply<false, true, T>(10, 10, 10, A.data(), 10, B.data(), 10, C.data(), 10, 1, 0);
	Math<T>::mAdd_mmt(10, 10, 10, A.data(), 10, B.data(), 10, D.data(), 10, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(C[i * 10 + j], D[i * 10 + j]), name + "::mAdd_mm failed!");
	
	addMatrixMultiply<true, true, T>(10, 10, 10, A.data(), 10, B.data(), 10, C.data(), 10, 1, 0);
	Math<T>::mAdd_mm(10, 10, 10, B.data(), 10, A.data(), 10, D.data(), 10, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual<T>(C[i * 10 + j], D[j * 10 + i]), name + "::mAdd_mm with both matrices transposed failed!");
}

void TestMath()
{
#if defined NN_REAL_T && !defined NN_IMPL
#define NN_STR(s) #s
	TestMathImpl<NN_REAL_T>(std::string("Math<") + NN_STR(NN_REAL_T) + ">");
#undef NN_STR
#elif !defined NN_IMPL
	TestMathImpl<double>("Math<double>");
	TestMathImpl<float>("Math<float>");
#endif
}

#endif
