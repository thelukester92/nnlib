#ifndef ONEHOT_H
#define ONEHOT_H

#include "module.h"

namespace nnlib
{

template <typename T = double>
class OneHot : public Module<T>
{
public:
	OneHot(size_t outs, size_t batch = 1)
	: m_inputBlame(batch, 1), m_outputs(batch, outs)
	{}
	
	virtual void resize(size_t inps, size_t outs, size_t bats) override
	{
		NNAssert(inps == 1, "OneHot can only accept a single input!");
		m_inputBlame.resize(bats, inps);
		m_outputs.resize(bats, outs);
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		
		m_outputs.fill(0.0);
		for(size_t i = 0; i < inputs.rows(); ++i)
			m_outputs(i, (size_t) inputs(i, 0)) = 1.0;
		
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		NNAssert(blame.rows() == m_outputs.rows(), "Incorrect batch size!");
		NNAssert(blame.cols() == m_outputs.cols(), "Incorrect blame size!");
		
		m_inputBlame.fill(0.0);
		for(size_t i = 0; i < inputs.rows(); ++i)
			m_inputBlame(i, 0) = blame(i, (size_t) inputs(i, 0));
		
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
	Matrix<T> m_inputBlame;		///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_outputs;		///< The output of this layer.
};

}

#endif
