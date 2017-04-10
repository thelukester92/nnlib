#ifndef IDENTITY_H
#define IDENTITY_H

#include "../module.h"

namespace nnlib
{

/// An activation function layer that passes input through as output (i.e. for residual connections).
template <typename T = double>
class Identity : public Module<T>
{
public:
	Identity(size_t inps = 0, size_t batch = 1) : m_inputBlame(batch, inps), m_outputs(batch, inps)
	{}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_outputs.rows() && inputs.cols() == m_outputs.cols(), "Incompatible input!");
		auto i = inputs.begin();
		auto j = m_outputs.begin(), end = m_outputs.end();
		for(; j != end; ++i, ++j)
			*j = *i;
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(blame.rows() == m_inputBlame.rows() && blame.cols() == m_inputBlame.cols(), "Incompatible blame!");
		auto i = blame.begin();
		auto j = m_inputBlame.begin(), end = m_inputBlame.end();
		for(; j != end; ++i, ++j)
			*j = *i;
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
