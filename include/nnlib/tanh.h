#ifndef TANH_H
#define TANH_H

namespace nnlib
{

/// Hyperbolic tangent activation function.
template <typename T = double>
class TanH
{
public:
	static T forward(const T &x)
	{
		return tanh(x);
	}
	
	static T backward(const T &x, const T &y)
	{
		return 1.0 - x * x;
	}
};

}

#endif
