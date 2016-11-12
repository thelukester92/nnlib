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
	Module(size_t inps, size_t outs) : m_output(outs), m_inputBlame(inps)
	{}
	
	Vector<T> &output()
	{
		return m_output;
	}
	
	Vector<T> &inputBlame()
	{
		return m_inputBlame;
	}
	
	size_t inputCount() const
	{
		return m_inputBlame.size();
	}
	
	size_t outputCount() const
	{
		return m_output.size();
	}
	
	/// Feed in an input vector and return a cached output vector.
	virtual Vector<T> &forward(const Vector<T> &input) = 0;
	
	/// Feed in an input and output blame (gradient) and return a cached input blame vector.
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &blame) = 0;
	
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
protected:
	Vector<T> m_output;
	Vector<T> m_inputBlame;
};

}

#endif
