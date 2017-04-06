#ifndef RMSPROP_H
#define RMSPROP_H

#include "optimizer.h"
#include "vector.h"

namespace nnlib
{

template <typename M, typename C>
class RMSProp : public Optimizer<M, C>
{
using Optimizer<M, C>::m_model;
using Optimizer<M, C>::m_critic;
public:
	using T = typename Optimizer<M, C>::T;
	
	RMSProp(M &model, C &critic, T lr = 0.01, T g = 0.9) :
		Optimizer<M, C>(model, critic),
		m_parameters(Vector<T>::flatten(model.parameters())),
		m_blame(Vector<T>::flatten(model.blame())),
		m_meanSquare(m_parameters.size(), 1.0),
		m_learningRate(lr), m_gamma(g)
	{}
	
	double learningRate() const
	{
		return m_learningRate;
	}
	
	RMSProp &learningRate(T lr)
	{
		m_learningRate = lr;
		return *this;
	}
	
	double gamma() const
	{
		return m_gamma;
	}
	
	RMSProp &gamma(T g)
	{
		m_gamma = g;
		return *this;
	}
	
	inline T isqrt(T number)
	{
		float x = number * 0.5f;
		float y = number;
		long i = *(long *) &y;
		i = 0x5f3759df - (i >> 1);
		y = *(float *) &i;
		return y * (1.5f - (x * y * y));
	}
	
	virtual void optimize(const Matrix<T> &inputs, const Matrix<T> &targets) override
	{
		m_blame.fill(0.0);
		m_model.backward(inputs, m_critic.backward(m_model.forward(inputs), targets));
		
		T scale = 1.0 - m_gamma;
		auto i = m_meanSquare.begin(), j = m_blame.begin(), end = m_blame.end();
		for(; j != end; ++i, ++j)
		{
			*i *= m_gamma;
			*i += scale * *j * *j;
			*j *= isqrt(*i);
		}
		
		m_parameters.addScaled(m_blame, m_learningRate / inputs.rows());
	}
	
private:
	Vector<T> m_parameters, m_blame, m_meanSquare;
	T m_learningRate, m_gamma;
};

}

#endif
