#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"

namespace nnlib
{

template <typename T>
class Linear : public Module<T>
{
public:
	Linear(size_t inps, size_t outs)
	: m_weights(outs, inps), m_bias(outs), m_weightsBlame(outs, inps), m_biasBlame(outs), m_inputBlame(inps), m_output(outs)
	{}
	
	Matrix<T> &weights()
	{
		return m_weights;
	}
	
	Vector<T> &bias()
	{
		return m_bias;
	}
	
	Matrix<T> &weightsBlame()
	{
		return m_weightsBlame;
	}
	
	Vector<T> &biasBlame()
	{
		return m_biasBlame;
	}
	
	virtual Vector<T> &forward(const Vector<T> &input) override
	{
		return m_output = m_weights * input + m_bias;
	}
	
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &blame) override
	{
		m_weightsBlame		= blame * input;
		m_biasBlame			= blame;
		return m_inputBlame	= blame * m_weights;
	}
	
private:
	// parameters
	Matrix<T> m_weights;
	Vector<T> m_bias;
	
	// buffers
	Matrix<T> m_weightsBlame;
	Vector<T> m_biasBlame;
	Vector<T> m_inputBlame;
	Vector<T> m_output;
};

}

#endif
