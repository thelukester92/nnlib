#ifndef SIN_H
#define SIN_H

#include "module.h"

namespace nnlib
{

/// Sinusoid activation function.
template <typename T = double>
class Sin
{
public:
	static T forward(const T &x)
	{
		return sin(x);
	}
	
	static T backward(const T &x, const T &y)
	{
		return cos(x);
	}
};

}

#endif
