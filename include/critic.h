#ifndef CRITIC_H
#define CRITIC_H

#include "tensor.h"

namespace nnlib
{

template <typename T>
class Critic
{
public:
	/// Feed in input and target vectors and return a cached error (loss) vector.
	virtual Vector<T> &forward(const Vector<T> &input, const Vector<T> &target) = 0;
	
	/// Feed in input and target vectors and return a cached blame (gradient) vector.
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &target) = 0;
	
	/// Get the error (loss) buffer.
	virtual Vector<T> &error() = 0;
	
	/// Get the blame (gradient) buffer.
	virtual Vector<T> &blame() = 0;
};

}

#endif
