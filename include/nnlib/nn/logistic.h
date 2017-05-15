#ifndef NN_LOGISTIC_H
#define NN_LOGISTIC_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Sigmoidal logistic activation function.
template <typename T = double>
class Logistic : public Map<T>
{
public:
	using Map<T>::Map;
	using Map<T>::forward;
	using Map<T>::backward;
	
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "logistic";
	}
	
	/// Single element forward.
	virtual T forward(const T &x) override
	{
		return 1.0 / (1.0 + exp(-x));
	}
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) override
	{
		return y * (1.0 - y);
	}
};

}

#endif
