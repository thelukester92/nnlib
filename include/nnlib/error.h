#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <sstream>
#include <exception>
#include <stdexcept>

/// Check for compiler support.
/// This is here because 1) it causes errors, and 2) this file is included everywhere.
/// \todo Determine if this check is better fitted elsewhere.
#if __cplusplus < 201103L
	#error C++11 is required! Use -std=c++11 if available.
#endif

/// \todo stack trace

namespace nnlib
{

/// Asserts that should never be optimized out.
#define NNHardAssert(x, m)										\
{																\
	if(!(x))													\
	{															\
		std::ostringstream oss;									\
		oss														\
			<< "Assert failed in " << __func__					\
			<< " at " __FILE__ << ":" << __LINE__ << std::endl	\
			<< "\t" << m << std::endl;							\
		throw std::runtime_error(oss.str());					\
	}															\
}

/// General asserts that can be optimized out after testing.
#ifndef OPTIMIZE
	#define NNAssert(x, m) NNHardAssert(x, m)
#else
	#define NNAssert(x, m) (void) 0
#endif

}

#endif
