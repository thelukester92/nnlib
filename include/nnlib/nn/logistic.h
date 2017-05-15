#ifndef NN_LOGISTIC_H
#define NN_LOGISTIC_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Sigmoidal logistic activation function.
class Logistic : public Map
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
		return "logistic";
	}
	
	/// Single element forward.
	virtual real_t forward(const real_t &x) override
	{
		return 1.0 / (1.0 + exp(-x));
	}
	
	/// Single element backward.
	virtual real_t backward(const real_t &x, const real_t &y) override
	{
		return y * (1.0 - y);
	}
};

}

#endif
