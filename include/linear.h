#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"

namespace nnlib
{

template <typename T>
class Linear : public Module<T>
{
using Module<T>::m_inputBlame;
using Module<T>::m_output;
public:
	Linear(size_t inps, size_t outs, size_t batchSize = 1)
	: Module<T>(inps, outs, batchSize), m_weights(outs, inps), m_bias(outs), m_weightsBlame(outs, inps), m_biasBlame(outs)
	{}
	
	/// Resize this module.
	virtual void resize(size_t inps, size_t outs, size_t batchSize) override
	{
		Module<T>::resize(inps, outs, batchSize);
		m_bias.resize(outs);
		m_weightsBlame.resize(outs, inps);
		m_biasBlame.resize(outs);
	}
	
	/// Feed in input vectors and return cached output vectors.
	virtual Matrix<T> &forward(const Matrix<T> &input) override
	{
		return m_output = input * ~m_weights + m_bias;
	}
	
	/// Feed in inputs and output blames (gradient) and return cached input blame vectors.
	virtual Matrix<T> &backward(const Matrix<T> &input, const Matrix<T> &blame) override
	{
		m_weightsBlame = ~blame * input;
		m_biasBlame = OpCollapse<Matrix<T>>(blame);
		return m_inputBlame = blame * m_weights;
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
