#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <sstream>
#include <exception>

namespace nnlib
{

#define NNLibAssert(x, m)										\
{																\
	if(!(x))													\
	{															\
		std::ostringstream oss;									\
		oss << "Assert failed in " << __func__					\
		    << " at " __FILE__ << ":" << __LINE__ << std::endl	\
		    << "\t" << m << std::endl;							\
		throw std::runtime_error(oss.str());					\
	}															\
}

/// General asserts that should only be optimized out after verifying the code works.
#ifndef OPTIMIZE
	#define NNAssert(x, m) NNLibAssert(x, m)
#else
	#define NNAssert(x, m) (void) 0
#endif

}

#endif
