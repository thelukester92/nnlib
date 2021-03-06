#ifndef NN_ACCEL_CPU
    #warning "You are not using any CPU acceleration! Define NN_ACCEL to use BLAS."
#else
#ifdef NN_REAL_T
#ifndef MATH_ALGEBRA_BLAS_TPP
#define MATH_ALGEBRA_BLAS_TPP

// in order to keep NN_REAL_T as the only required define,
// the following hack avoids needing "#if NN_REAL_T == double"
// by requiring a different file based on that define.
// if NN_ACCEL_CPU is enabled but NN_REAL_T is not double or float,
// this will cause a file not found error on the include directive.

#define  NN_TEMP_0(x)     #x
#define  NN_TEMP_1(x,y,z) NN_TEMP_0(x##y##z)
#define  NN_TEMP_2(x,y,z) NN_TEMP_1(x,y,z)
#include NN_TEMP_2(algebra_blas_, NN_REAL_T, _impl.tpp)
#undef NN_TEMP_0
#undef NN_TEMP_1
#undef NN_TEMP_2

#endif
#endif
#endif
