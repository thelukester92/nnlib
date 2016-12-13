#ifndef TANH_H
#define TANH_H

#include "module.h"

namespace nnlib
{

template <typename T>
class TanH : public Module<T>
{
public:
	TanH(size_t size, size_t batch)
	: m_inputBlame(batch, size), m_outputs(batch, size)
	{}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		auto i = inputs.begin();
		auto j = m_outputs.begin(), end = m_outputs.end();
		for(; j != end; ++i, ++j)
			*j = tanh(*i);
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		auto i = inputs.begin(), k = blame.begin();
		auto j = m_inputBlame.begin(), end = m_inputBlame.end();
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
