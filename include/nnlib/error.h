#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <sstream>
#include <exception>

namespace nnlib
{

/// Warnings that act as asserts in DEBUG mode.
#define NNWarn(x, m)											\
{																\
	if(!(x))													\
	{															\
		std::cerr												\
			<< "Warning in " << __func__						\
			<< " at " __FILE__ << ":" << __LINE__ << std::endl	\
			<< "\t" << m << std::endl;							\
	}															\
}

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

/// Turn warns into asserts in debug mode.
#ifdef DEBUG
	#undef NNWarn
	#define NNWarn(x, m) NNHardAssert(x, m)
#endif

/// Turn warns into noops in optimize mode.
#ifdef OPTIMIZE
	#undef NNWarn
	#define NNWarn(x, m) (void) 0
#endif

}

#endif
