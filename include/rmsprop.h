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
	
	RMSProp(M &model, C &critic, T lr = 0.01, T g = 0.9)
	: Optimizer<M, C>(model, critic),
	  m_parameters(model.parameters()), m_blame(model.blame()),
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
	
	virtual void optimize(const Matrix<T> &inputs, const Matrix<T> &targets) override
	{
		static const T epsilon = 1e-9;
		
		m_blame.fill(0.0);
		m_model.backward(inputs, m_critic.backward(m_model.forward(inputs), targets));
		
		T scale = 1.0 - m_gamma;
		auto i = m_meanSquare.begin(), j = m_blame.begin(), end = m_blame.end();
		for(; j != end; ++i, ++j)
		{
			*i *= m_gamma;
			*i += scale * *j * *j;
			
			/// \todo fast inverse square root instead of this
			*j /= sqrt(*i) + epsilon;
		}
		
		m_parameters.addScaled(m_blame, m_learningRate);
	}
	
private:
	Vector<T> m_parameters, m_blame, m_meanSquare;
	T m_learningRate, m_gamma;
};

}

#endif
