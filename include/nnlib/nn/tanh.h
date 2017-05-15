#ifndef NN_TANH_H
#define NN_TANH_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Hyperbolic tangent activation function.
class TanH : public Map
{
public:
	using Map::Map;
	using Map::forward;
	using Map::backward;
	
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "tanh";
	}
	
	/// Single element forward.
	virtual real_t forward(const real_t &x) override
	{
		return tanh(x);
	}
	
	/// Single element backward.
	virtual real_t backward(const real_t &x, const real_t &y) override
	{
		return 1.0 - y * y;
	}
};

}

#endif
