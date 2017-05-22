#ifndef TEST_MATH_BLAS
#define TEST_MATH_BLAS

#ifdef HAS_BLAS

#include "nnlib/math/math_blas.h"
#include "test_math_base.h"
using namespace nnlib;

void TestMathBLAS()
{
	std::vector<double> x(10), y(10);
	std::vector<double> A(100), B(100), C(100), D(100);
	
	std::fill(x.begin(), x.end(), 0.5);
	MathBLAS<double>::vFill(y.data(), y.size(), 1, 0.5);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), "MathBLAS<>::vFill with stride = 1 failed!");
	
	for(size_t i = 0; i < x.size(); i += 2)
		x[i] = 3.14;
	MathBLAS<double>::vFill(y.data(), y.size() / 2, 2, 3.14);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), "MathBLAS<>::vFill with stride = 2 failed!");
	
	for(size_t i = 0; i < x.size(); ++i)
		x[i] *= 2;
	MathBLAS<double>::vScale(y.data(), y.size(), 1, 2);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), "MathBLAS<>::vScale with stride = 1 failed!");
	
	for(size_t i = 0; i < x.size(); i += 2)
		x[i] *= 0.75;
	MathBLAS<double>::vScale(y.data(), y.size() / 2, 2, 0.75);
	NNHardAssert(std::equal(x.begin(), x.end(), y.begin()), "MathBLAS<>::vScale with stride = 2 failed!");
	
	std::fill(A.begin(), A.end(), 0.5);
	MathBLAS<double>::mFill(B.data(), 10, 10, 10, 0.5);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), "MathBLAS<>::mFill with ld == cols failed!");
	
	for(size_t i = 0; i < 10; i += 2)
		for(size_t j = 0; j < 10; ++j)
			A[i * 10 + j] = 3.14;
	MathBLAS<double>::mFill(B.data(), 5, 10, 20, 3.14);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), "MathBLAS<>::mFill with ld != cols failed!");
	
	for(size_t i = 0; i < A.size(); ++i)
		A[i] *= 2;
	MathBLAS<double>::mScale(B.data(), 10, 10, 10, 2);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), "MathBLAS<>::mScale with ld == cols failed!");
	
	for(size_t i = 0; i < 10; i += 2)
		for(size_t j = 0; j < 10; ++j)
			A[i * 10 + j] *= 0.75;
	MathBLAS<double>::mScale(B.data(), 5, 10, 20, 0.75);
	NNHardAssert(std::equal(A.begin(), A.end(), B.begin()), "MathBLAS<>::mScale with ld != cols failed!");
	
	std::fill(y.begin(), y.end(), 0.12345);
	MathBLAS<double>::vAdd_v(x.data(), x.size(), 1, y.data(), 1);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual(x[i] + 0.12345, y[i]), "MathBLAS<>::vAdd_v with stride = 1 failed!");
	
	MathBLAS<double>::vAdd_v(y.data(), y.size() / 2, 2, y.data(), 2);
	for(size_t i = 0; i < x.size(); ++i)
	{
		if(i % 2 == 0)
		{
			NNHardAssert(almostEqual(x[i] + 0.12345 + x[i] + 0.12345, y[i]), "MathBLAS<>::vAdd_v with stride = 2 failed!");
		}
		else
		{
			NNHardAssert(almostEqual(x[i] + 0.12345, y[i]), "MathBLAS<>::vAdd_v with stride = 2 failed!");
		}
	}
	
	std::fill(y.begin(), y.end(), 0.12345);
	MathBLAS<double>::vAdd_v(x.data(), x.size(), 1, y.data(), 1, 0.5);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual(0.5 * x[i] + 0.12345, y[i]), "MathBLAS<>::vAdd_v with alpha != 1 failed!");
	
	std::fill(y.begin(), y.end(), 0.12345);
	MathBLAS<double>::vAdd_v(x.data(), x.size(), 1, y.data(), 1, 1, 0.75);
	for(size_t i = 0; i < x.size(); ++i)
		NNHardAssert(almostEqual(x[i] + 0.75 * 0.12345, y[i]), "MathBLAS<>::vAdd_v with beta != 1 failed!");
	
	std::fill(A.begin(), A.end(), 0);
	MathBLAS<double>::mAdd_vv(x.data(), x.size(), 1, y.data(), y.size(), 1, A.data(), 10);
	for(size_t i = 0; i < x.size(); ++i)
		for(size_t j = 0; j < y.size(); ++j)
			NNHardAssert(almostEqual(A[i * 10 + j], x[i] * y[j]), "MathBLAS<>::mAdd_vv with stride = 1 failed!");
	
	MathBLAS<double>::mAdd_vv(x.data(), x.size() / 2, 2, y.data(), y.size() / 2, 2, A.data(), 10);
	for(size_t i = 0; i < x.size(); ++i)
	{
		for(size_t j = 0; j < y.size(); ++j)
		{
			if(i < x.size() / 2 && j < y.size() / 2)
			{
				NNHardAssert(almostEqual(A[i * 10 + j], x[i] * y[j] + x[2*i] * y[2*j]), "MathBLAS<>::mAdd_vv with stride = 2 failed!");
			}
			else
			{
				NNHardAssert(almostEqual(A[i * 10 + j], x[i] * y[j]), "MathBLAS<>::mAdd_vv with stride = 2 failed!");
			}
		}
	}
	
	std::fill(y.begin(), y.end(), 0);
	MathBLAS<double>::vAdd_mv(A.data(), 10, 10, 10, x.data(), 1, y.data(), 1);
	for(size_t i = 0; i < 10; ++i)
	{
		double value = 0.0;
		for(size_t j = 0; j < 10; ++j)
		{
			value += A[i * 10 + j] * x[j];
		}
		NNHardAssert(almostEqual(y[i], value), "MathBLAS<>::vAdd_mv failed!");
	}
	
	std::fill(y.begin(), y.end(), 0);
	MathBLAS<double>::vAdd_mtv(A.data(), 10, 10, 10, x.data(), 1, y.data(), 1);
	for(size_t i = 0; i < 10; ++i)
	{
		double value = 0.0;
		for(size_t j = 0; j < 10; ++j)
		{
			value += A[j * 10 + i] * x[j];
		}
		NNHardAssert(almostEqual(y[i], value), "MathBLAS<>::vAdd_mtv failed!");
	}
	
	std::fill(B.begin(), B.end(), 10);
	MathBLAS<double>::mAdd_m(A.data(), 10, 10, 10, B.data(), 10);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual(B[i * 10 + j], A[i * 10 + j] + 10), "MathBLAS<>::mAdd_m failed!");
	
	std::fill(B.begin(), B.end(), 10);
	MathBLAS<double>::mAdd_mt(A.data(), 10, 10, 10, B.data(), 10);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual(B[i * 10 + j], A[j * 10 + i] + 10), "MathBLAS<>::mAdd_mt failed!");
	
	std::fill(C.begin(), C.end(), 0);
	addMatrixMultiply<false, false>(10, 10, 10, A.data(), 10, B.data(), 10, C.data(), 10, 1, 0);
	MathBLAS<double>::mAdd_mm(10, 10, 10, A.data(), 10, B.data(), 10, D.data(), 10, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual(C[i * 10 + j], D[i * 10 + j]), "MathBLAS<>::mAdd_mm failed!");
	
	addMatrixMultiply<true, false>(10, 10, 10, A.data(), 10, B.data(), 10, C.data(), 10, 1, 0);
	MathBLAS<double>::mAdd_mtm(10, 10, 10, A.data(), 10, B.data(), 10, D.data(), 10, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual(C[i * 10 + j], D[i * 10 + j]), "MathBLAS<>::mAdd_mm failed!");
	
	addMatrixMultiply<false, true>(10, 10, 10, A.data(), 10, B.data(), 10, C.data(), 10, 1, 0);
	MathBLAS<double>::mAdd_mmt(10, 10, 10, A.data(), 10, B.data(), 10, D.data(), 10, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual(C[i * 10 + j], D[i * 10 + j]), "MathBLAS<>::mAdd_mm failed!");
	
	addMatrixMultiply<true, true>(10, 10, 10, A.data(), 10, B.data(), 10, C.data(), 10, 1, 0);
	MathBLAS<double>::mAdd_mm(10, 10, 10, B.data(), 10, A.data(), 10, D.data(), 10, 1, 0);
	for(size_t i = 0; i < 10; ++i)
		for(size_t j = 0; j < 10; ++j)
			NNHardAssert(almostEqual(C[i * 10 + j], D[j * 10 + i]), "MathBLAS<>::mAdd_mm with both matrices transposed failed!");
}

#endif

#endif
