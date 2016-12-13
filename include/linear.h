#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"

namespace nnlib
{

template <typename T>
class Linear : public Module<T>
{
public:
	Linear(size_t inps, size_t outs, size_t batch)
	: m_addBuffer(batch, 1),
	  m_bias(outs), m_weights(outs, inps),
	  m_biasBlame(outs), m_weightsBlame(outs, inps),
	  m_inputBlame(batch, inps), m_outputs(batch, outs)
	{}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		Matrix<T>::multiply(inputs, m_weights, m_outputs, false, true);
		Matrix<T>::addOuterProduct(m_addBuffer, m_bias, m_outputs);
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		Matrix<T>::multiply(blame, inputs, m_weightsBlame, true, false);
		Matrix<T>::multiply(blame, m_addBuffer, m_biasBlame);
		Matrix<T>::multiply(blame, m_weights, m_inputBlame);
		return m_inputBlame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_outputs;
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
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
	Vector<T> m_addBuffer;		///< A vector of 1s to quickly evaluate bias.
	Vector<T> m_bias;			///< The bias; adding constants to outputs.
	Matrix<T> m_weights;		///< The parameters.
	Vector<T> m_biasBlame;		///< Gradient of the error w.r.t. the bias.
	Matrix<T> m_weightsBlame;	///< Gradient of the error w.r.t. the weights.
	Matrix<T> m_inputBlame;		///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_outputs;		///< The output of this layer.
};

}

#endif
