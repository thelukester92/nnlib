#ifndef LOGISTIC_H
#define LOGISTIC_H

#include "module.h"

namespace nnlib
{

/// Logistic activation function.
template <typename T = double>
class Logistic
{
public:
	static T forward(const T &x)
	{
		return 1.0 / (1.0 + exp(-x));
	}
	
	static T backward(const T &x, const T &y)
	{
		return y * (1.0 - y);
	}
};

}

#endif
