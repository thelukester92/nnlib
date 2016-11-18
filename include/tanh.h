#ifndef TANH_H
#define TANH_H

#include "module.h"
#include <cmath>

namespace nnlib
{

template <typename T>
class Tanh : public Module<T>
{
using Module<T>::m_inputBlame;
using Module<T>::m_output;
public:
	Tanh(size_t inps, size_t batchSize = 1) : Module<T>(inps, inps, batchSize)
	{}
	
	/// Feed in input vectors and return cached output vectors.
	virtual Matrix<T> &forward(const Matrix<T> &input) override
	{
		size_t n = input.size();
		NNAssert(n == m_inputBlame.size(), "Incompatible input!");
		for(size_t i = 0; i < n; ++i)
			m_output[i] = tanh(input[i]);
		return m_output;
	}
	
	/// Feed in inputs and output blames (gradient) and return cached input blame vectors.
	virtual Matrix<T> &backward(const Matrix<T> &input, const Matrix<T> &blame) override
	{
		size_t n = input.size();
		NNAssert(n == m_inputBlame.size(), "Incompatible input!");
		NNAssert(n == blame.size(), "Incompatible blame!");
		for(size_t i = 0; i < n; ++i)
			m_inputBlame[i] = blame[i] * (1.0 - m_output[i] * m_output[i]);
		return m_inputBlame;
	}
};

}

#endif
