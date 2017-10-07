#ifndef NN_LOGISTIC_HPP
#define NN_LOGISTIC_HPP

#include <math.h>
#include "map.hpp"

namespace nnlib
{

/// Sigmoidal logistic activation function.
template <typename T = double>
class Logistic : public Map<T>
{
public:
	Logistic() {}
	Logistic(const Serialized &) {}
	Logistic(const Logistic &) {}
	Logistic &operator=(const Logistic &) { return *this; }
	
	/// Single element forward.
	virtual T forwardOne(const T &x) override
	{
		return 1.0 / (1.0 + exp(-x));
	}
	
	/// Single element backward.
	virtual T backwardOne(const T &x, const T &y) override
	{
		return y * (1.0 - y);
	}
};

}

NNRegisterType(Logistic, Module);

#endif
