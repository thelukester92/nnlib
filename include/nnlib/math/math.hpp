#ifndef MATH_MATH_HPP
#define MATH_MATH_HPP

#ifdef ACCELERATE_BLAS
	#include "math_blas.hpp"
#else
	#include "math_base.hpp"
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
