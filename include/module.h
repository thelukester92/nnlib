#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"

namespace nnlib
{

/// \todo what if the module feeds forward a Matrix instead of a Vector?
/// \todo Matrix-Matrix feed-in and Matrix-Vector feed-in.
/// \todo inputBlame and output protected parameters.

template <typename T>
class Module
{
public:
	virtual ~Module() {}
	
	size_t inputCount() const
	{
		return inputBlame().size();
	}
	
	size_t outputCount() const
	{
		return output().size();
	}
	
	const Vector<T> &inputBlame() const
	{
		return const_cast<Module *>(this)->inputBlame();
	}
	
	const Vector<T> &output() const
	{
		return const_cast<Module *>(this)->output();
	}
	
	/// Feed in an input vector and return a cached output vector.
	virtual Vector<T> &forward(const Vector<T> &input) = 0;
	
	/// Feed in an input and output blame (gradient) and return a cached input blame vector.
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &blame) = 0;
	
	/// Get the input blame (gradient) buffer.
	virtual Vector<T> &inputBlame() = 0;
	
	/// Get the output buffer.
	virtual Vector<T> &output() = 0;
	
	/// Return pointers to all parameters (i.e. for flattening).
	virtual Vector<Tensor<T> *> parameters()
	{
		return Vector<Tensor<T> *>(0);
	}
	
	/// Return pointers to parameter blame buffers (i.e. for flattening).
	virtual Vector<Tensor<T> *> blame()
	{
		return Vector<Tensor<T> *>(0);
	}
};

}

#endif
