#ifndef MATH_H
#define MATH_H

#ifdef ACCELERATE_BLAS
	#include "math_blas.h"
#else
	#include "math_base.h"
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
