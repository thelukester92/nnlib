#ifndef MATH_H
#define MATH_H

#ifdef ACCELERATE_BLAS
	#include "math_blas.h"
#else
	#include "math_base.h"
	#warning You are not using any acceleration!
	#ifdef __APPLE__
		#warning Use -framework Accelerate -DACCELERATE_BLAS for significant speedup.
	#else
		#warning Use -lopenblas -DACCELERATE_BLAS for significant speedup.
	#endif
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
