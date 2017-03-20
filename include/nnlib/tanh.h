#ifndef TANH_H
#define TANH_H

#include "module.h"

namespace nnlib
{

/// An activation function layer that applies tanh to each input.
template <typename T = double>
class TanH : public Module<T>
{
public:
	TanH(size_t size = 0, size_t batch = 1)
	: m_inputBlame(batch, size), m_outputs(batch, size)
	{}
	
	virtual void resize(size_t inps) override
	{
		Module<T>::resize(inps, inps, m_outputs.rows());
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		
		auto i = inputs.begin();
		auto j = m_outputs.begin(), end = m_outputs.end();
		for(; j != end; ++i, ++j)
			*j = tanh(*i);
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		NNAssert(blame.rows() == m_outputs.rows(), "Incorrect batch size!");
		NNAssert(blame.cols() == m_outputs.cols(), "Incorrect blame size!");
		auto k = blame.begin();
		auto i = m_outputs.begin(), j = m_inputBlame.begin(), end = m_inputBlame.end();
		for(; j != end; ++i, ++j, ++k)
			*j = *k * (1.0 - *i * *i);
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
	
private:
	Matrix<T> m_inputBlame;	///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_outputs;	///< The output of this layer.
};

}

#endif
