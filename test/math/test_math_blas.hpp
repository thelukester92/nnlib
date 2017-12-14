#ifndef TEST_MATH_BLAS
#define TEST_MATH_BLAS

#ifdef NN_ACCEL
	#include "nnlib/math/math_blas.hpp"
#endif

#include "test_math_base.hpp"
using namespace nnlib;

void TestMathBLAS()
{
#ifdef NN_REAL_T
#define NN_STR(s) #s
	TestMath<MathBLAS, NN_REAL_T>(std::string("MathBLAS<") + NN_STR(NN_REAL_T) + ">");
#undef NN_STR
#else
	TestMath<MathBLAS, double>("MathBLAS<double>");
	TestMath<MathBLAS, float>("MathBLAS<float>");
#endif
}

#endif
