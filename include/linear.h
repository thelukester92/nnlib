#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"

namespace nnlib
{

template <typename T>
class Linear : public Module<T>
{
using Module<T>::m_output;
using Module<T>::m_inputBlame;
public:
	Linear(size_t inps, size_t outs)
	: Module<T>(inps, outs), m_weights(outs, inps), m_bias(outs), m_weightsBlame(outs, inps), m_biasBlame(outs)
	{}
	
	/// Feed in an input vector and return a cached output vector.
	virtual Vector<T> &forward(const Vector<T> &input) override
	{
		Assert(input.size() == m_weights.cols(), "Incompatible input!");
		return m_output = m_weights * input + m_bias;
	}
	
	/// Feed in an input and output blame (gradient) and return a cached input blame vector.
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &blame) override
	{
		Assert(input.size() == m_weights.cols(), "Incompatible input!");
		Assert(blame.size() == m_weights.rows(), "Incompatible blame!");
		m_weightsBlame		= blame * input;
		m_biasBlame			= blame;
		return m_inputBlame	= blame * m_weights;
	}
	
	/// Return pointers to all parameters (i.e. for flattening).
	virtual Vector<Tensor<T> *> parameters() override
	{
		return { &m_weights, &m_bias };
	}
	
	/// Return pointers to parameter blame buffers (i.e. for flattening).
	virtual Vector<Tensor<T> *> blame() override
	{
		return { &m_weightsBlame, &m_biasBlame };
	}
	
private:
	// parameters
	Matrix<T> m_weights;
	Vector<T> m_bias;
	
	// buffers
	Matrix<T> m_weightsBlame;
	Vector<T> m_biasBlame;
};

}

#endif
