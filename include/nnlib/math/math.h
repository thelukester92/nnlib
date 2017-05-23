#ifndef MATH_H
#define MATH_H

#ifdef ACCELERATE_BLAS
	#include "math_blas.h"
	#pragma message "You are accelerating with BLAS!"
#else
	#include "math_base.h"
	#warning "You are not using any acceleration! Use BLAS for significant speedup."
#endif

namespace nnlib
{

#ifdef ACCELERATE_BLAS
	template <typename T = double>
	using Math = MathBLAS<T>;
#else
	template <typename T = double>
	using Math = MathBase<T>;
#endif

}

#endif
