#ifndef SIN_H
#define SIN_H

#include "module.h"

namespace nnlib
{

template <typename T = double>
class Sin : public Module<T>
{
public:
	Sin(size_t size = 0, size_t batch = 1)
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
			*j = sin(*i);
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		NNAssert(blame.rows() == m_outputs.rows(), "Incorrect batch size!");
		NNAssert(blame.cols() == m_outputs.cols(), "Incorrect blame size!");
		auto k = blame.begin();
		auto i = inputs.begin();
		auto j = m_inputBlame.begin(), end = m_inputBlame.end();
		for(; j != end; ++i, ++j, ++k)
			*j = *k * cos(*i);
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