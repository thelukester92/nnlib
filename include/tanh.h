#ifndef TANH_H
#define TANH_H

#include "module.h"
#include <cmath>

namespace nnlib
{

template <typename T>
class Tanh : public Module<T>
{
public:
	Tanh(size_t inps) : m_inputBlame(inps), m_output(inps)
	{}
	
	/// Feed in an input vector and return a cached output vector.
	virtual Vector<T> &forward(const Vector<T> &input) override
	{
		size_t n = input.size();
		Assert(n == m_inputBlame.size(), "Incompatible input!");
		for(size_t i = 0; i < n; ++i)
			m_output[i] = tanh(input[i]);
		return m_output;
	}
	
	/// Feed in an input and output blame (gradient) and return a cached input blame vector.
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &blame) override
	{
		size_t n = input.size();
		Assert(n == m_inputBlame.size(), "Incompatible input!");
		Assert(n == blame.size(), "Incompatible blame!");
		for(size_t i = 0; i < n; ++i)
			m_inputBlame[i] = 1.0 - m_output[i] * m_output[i];
		return m_inputBlame;
	}
	
	/// Get the input blame (gradient) buffer.
	virtual Vector<T> &inputBlame() override
	{
		return m_inputBlame;
	}
	
	/// Get the output buffer.
	virtual Vector<T> &output() override
	{
		return m_output;
	}

private:
	Vector<T> m_inputBlame;
	Vector<T> m_output;
};

}

#endif
