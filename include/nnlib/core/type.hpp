#ifndef CORE_TYPE_HPP
#define CORE_TYPE_HPP

#include <cstddef>

/// Check for compiler support.
/// This is here because this file is included everywhere.
#if __cplusplus < 201103L
	#error C++11 is required! Use -std=c++11 if available.
#endif

/// Check for a predefined real type.
#if !defined NN_REAL_T && !defined NN_HEADER_ONLY
	#define NN_REAL_T double
#endif

#endif
