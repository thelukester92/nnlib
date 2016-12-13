#ifndef SSE_H
#define SSE_H

#include "critic.h"

namespace nnlib
{

template <typename T>
class SSE : public Critic<T>
{
public:
	SSE(size_t size, size_t batch) : m_output(batch, size), m_blame(batch, size) {}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs, const Matrix<T> &targets) override
	{
		auto i = inputs.begin(), j = targets.begin();
		auto k = m_output.begin(), end = m_output.end();
		for(; k != end; ++i, ++j, ++k)
			*k = (*j - *i) * (*j - *i);
		return m_output;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &targets) override
	{
		auto i = inputs.begin(), j = targets.begin();
		auto k = m_blame.begin(), end = m_blame.end();
		for(; k != end; ++i, ++j, ++k)
			*k = *j - *i;
		return m_blame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_output;
	}
	
	virtual Matrix<T> &blame() override
	{
		return m_blame;
	}
private:
	Matrix<T> m_output, m_blame;
};

}

#endif
