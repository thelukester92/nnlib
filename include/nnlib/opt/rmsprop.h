#ifndef RMSPROP_H
#define RMSPROP_H

#include "optimizer.h"

namespace nnlib
{

class RMSProp : public Optimizer
{
using Optimizer::m_model;
using Optimizer::m_critic;
public:
	RMSProp(Module &model, Critic &critic) :
		Optimizer(model, critic),
		m_parameters(model.parameters()),
		m_grads(model.grad()),
		m_learningRate(0.001),
		m_gamma(0.9)
	{
		m_variance.resize(m_grads.size()).fill(0.0);
	}
	
	RMSProp &learningRate(real_t learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	real_t learningRate() const
	{
		return m_learningRate;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual RMSProp &step(const Tensor &input, const Tensor &target) override
	{
		// calculate gradient
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		
		for(size_t i = 0, end = m_grads.size(); i != end; ++i)
		{
			// update variance
			m_variance(i) = m_gamma * m_variance(i) + (1 - m_gamma) * m_grads(i) * m_grads(i);
			
			// update parameters
			m_parameters(i) -= m_learningRate * m_grads(i) / (sqrt(m_variance(i)) + 1e-8);
		}
		
		return *this;
	}
	
private:
	Tensor &m_parameters;
	Tensor &m_grads;
	Tensor m_variance;
	real_t m_learningRate;
	real_t m_gamma;
};

}

#endif
