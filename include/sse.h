#ifndef SSE_H
#define SSE_H

#include "critic.h"

namespace nnlib
{

template <typename T>
class SSE : public Critic<T>
{
public:
	SSE(size_t batch, size_t size) : m_blame(batch, size) {}
	
	virtual void calculateBlame(const Matrix<T> &inputs, const Matrix<T> &targets) override
	{
		auto i = inputs.begin(), j = targets.begin();
		auto k = m_blame.begin(), end = m_blame.end();
		for(; k != end; ++i, ++j, ++k)
			*k = *j - *i;
	}
	
	virtual Matrix<T> &blame() override
	{
		return m_blame;
	}
private:
	Matrix<T> m_blame;
};

}

#endif
