#ifndef MATH_MATH_HPP
#define MATH_MATH_HPP

#include "math_base.hpp"
#include "math_blas_double.hpp"
// #include "math_blas_float.hpp"

namespace nnlib
{

template <typename T = double>
using Math = MathBLAS<T>;

}

#endif
