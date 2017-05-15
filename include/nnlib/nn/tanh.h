#ifndef NN_TANH_H
#define NN_TANH_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Hyperbolic tangent activation function.
template <typename T = double>
class TanH : public Map<T>
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
		return "tanh";
	}
	
	/// Single element forward.
	virtual T forward(const T &x) override
	{
		return tanh(x);
	}
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) override
	{
		return 1.0 - y * y;
	}
};

}

#endif
