#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"

namespace nnlib
{

/// \todo what if the module feeds forward a Matrix instead of a Vector?

template <typename T>
class Module
{
public:
	virtual Vector<T> &forward(const Vector<T> &input) = 0;
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &blame) = 0;
};

}

#endif
