#ifndef TEST_MATH_BLAS
#define TEST_MATH_BLAS

#ifdef HAS_BLAS

#include "nnlib/math/math_blas.h"
#include "test_math_base.h"
using namespace nnlib;

void TestMathBLAS()
{
	TestMath<MathBLAS, double>("MathBLAS<double>");
	TestMath<MathBLAS, float>("MathBLAS<float>");
}

#endif

#endif
