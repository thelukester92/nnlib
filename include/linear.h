#ifndef LINEAR_H
#define LINEAR_H

#include <iostream>
#include "module.h"

namespace nnlib
{

template <typename T>
class Linear : public Module<T>
{
public:
	Linear(size_t inps, size_t outs, size_t batch)
	: m_bias(outs), m_addBuffer(batch, 1), m_weights(outs, inps), m_biasBlame(outs), m_weightsBlame(outs, inps), m_inputBlame(batch, inps), m_outputs(batch, outs) {}
	
	virtual void forward(const Matrix<T> &inputs) override
	{
		Matrix<T>::multiply(inputs, m_weights, m_outputs, false, true);
		Matrix<T>::addOuterProduct(m_addBuffer, m_bias, m_outputs);
	}
	
	virtual void backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		/// \todo test this and add bias
		Matrix<T>::multiply(blame, inputs, m_weightsBlame, true, false);
		Matrix<T>::multiply(blame, m_weights, m_inputBlame);
	}
	
	virtual Matrix<T> &output() override
	{
		return m_outputs;
	}
	
	virtual Vector<Tensor<T> *> parameters() override
	{
		return { &m_bias, &m_weights };
	}
	
	virtual Vector<Tensor<T> *> blame() override
	{
		return { &m_biasBlame, &m_weightsBlame };
	}
	
private:
	Vector<T> m_bias, m_addBuffer;
	Matrix<T> m_weights;
	
	Vector<T> m_biasBlame;
	Matrix<T> m_weightsBlame;
	
	Matrix<T> m_inputBlame;
	Matrix<T> m_outputs;
};

}

#endif
