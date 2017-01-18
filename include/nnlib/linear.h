#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"
#include "random.h"

namespace nnlib
{

template <typename T = double>
class Linear : public Module<T>
{
public:
	Linear(size_t inps, size_t outs, size_t batch = 1)
	: m_addBuffer(batch, 1),
	  m_bias(outs), m_weights(outs, inps),
	  m_biasBlame(outs), m_weightsBlame(outs, inps),
	  m_inputBlame(batch, inps), m_outputs(batch, outs)
	{
		resetWeights();
	}
	
	Linear(size_t outs)
	: m_addBuffer(1, 1),
	  m_bias(outs), m_weights(outs, 0),
	  m_biasBlame(outs), m_weightsBlame(outs, 0),
	  m_inputBlame(1, 0), m_outputs(1, outs)
	{
		resetWeights();
	}
	
	Vector<T> &bias()
	{
		return m_bias;
	}
	
	Matrix<T> &weights()
	{
		return m_weights;
	}
	
	void resetWeights()
	{
		for(double &val : m_weights)
			val = Random<T>::normal(0, 1, 1);
		for(double &val : m_bias)
			val = Random<T>::normal(0, 1, 1);
	}
	
	virtual void resize(size_t inps, size_t outs, size_t bats) override
	{
		m_addBuffer.resize(bats).fill(1);
		m_inputBlame.resize(bats, inps);
		m_outputs.resize(bats, outs);
		m_bias.resize(outs);
		m_weights.resize(outs, inps);
		m_biasBlame.resize(outs);
		m_weightsBlame.resize(outs, inps);
		resetWeights();
	}
	
	virtual void batch(size_t size) override
	{
		m_addBuffer.resize(size).fill(1);
		m_inputBlame.resize(size, m_inputBlame.cols());
		m_outputs.resize(size, m_outputs.cols());
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		Matrix<T>::multiply(inputs, m_weights, m_outputs, false, true);
		Matrix<T>::addOuterProduct(m_addBuffer, m_bias, m_outputs);
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		NNAssert(blame.rows() == m_outputs.rows(), "Incorrect batch size!");
		NNAssert(blame.cols() == m_outputs.cols(), "Incorrect blame size!");
		Matrix<T>::multiply(blame, inputs, m_weightsBlame, true, false, 1, 1);
		Matrix<T>::multiply(blame, m_addBuffer, m_biasBlame, false, 1, 1);
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
