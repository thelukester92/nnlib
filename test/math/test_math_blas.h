#ifndef TEST_MATH_BLAS
#define TEST_MATH_BLAS

#ifdef ACCELERATE_BLAS
	#include "nnlib/math/math_blas.h"
#endif

#include "test_math_base.h"
using namespace nnlib;

void TestMathBLAS()
{
#ifdef ACCELERATE_BLAS
	TestMath<MathBLAS, double>("MathBLAS<double>");
	TestMath<MathBLAS, float>("MathBLAS<float>");
#endif
}

#endif
