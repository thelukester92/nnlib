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

#ifdef DEBUG
	#define DebugAssert(x, m) NNLibAssert(x, m)
#else
	#define DebugAssert(x, m) (void) 0
#endif

#ifndef OPTIMIZE
	#define Assert(x, m) NNLibAssert(x, m)
#else
	#define Assert(x, m) (void) 0
#endif

}

#endif
