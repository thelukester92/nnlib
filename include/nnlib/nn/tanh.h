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
	TanH() {}
	TanH(const Serialized &) {}
	TanH(const TanH &) {}
	TanH &operator=(const TanH &) { return *this; }
	
	/// Single element forward.
	virtual T forwardOne(const T &x) override
	{
		return tanh(x);
	}
	
	/// Single element backward.
	virtual T backwardOne(const T &x, const T &y) override
	{
		return 1.0 - y * y;
	}
};

}

NNRegisterType(TanH, Module);

#endif
