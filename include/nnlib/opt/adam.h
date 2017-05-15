#ifndef ADAM_H
#define ADAM_H

#include "optimizer.h"

namespace nnlib
{

class Adam : public Optimizer
{
using Optimizer::m_model;
using Optimizer::m_critic;
public:
	Adam(Module &model, Critic &critic) :
		Optimizer(model, critic),
		m_parameters(model.parameters()),
		m_grads(model.grad()),
		m_learningRate(0.001),
		m_beta1(0.9),
		m_beta2(0.999),
		m_normalize1(1),
		m_normalize2(1),
		m_steps(0)
	{
		m_mean.resize(m_grads.size()).fill(0.0);
		m_variance.resize(m_grads.size()).fill(0.0);
	}
	
	void reset()
	{
		m_steps = 0;
		m_normalize1 = 1;
		m_normalize2 = 1;
	}
	
	Adam &learningRate(real_t learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	real_t learningRate() const
	{
		return m_learningRate;
	}
	
	Adam &beta1(real_t beta1)
	{
		m_beta1 = beta1;
		return *this;
	}
	
	real_t beta1() const
	{
		return m_beta1;
	}
	
	Adam &beta2(real_t beta2)
	{
		m_beta2 = beta2;
		return *this;
	}
	
	real_t beta2() const
	{
		return m_beta2;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual Adam &step(const Tensor &input, const Tensor &target) override
	{
		m_normalize1 *= m_beta1;
		m_normalize2 *= m_beta2;
		++m_steps;
		
		real_t lr = m_learningRate / (1 - m_normalize1) * sqrt(1 - m_normalize2);
		
		// calculate gradient
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		
		for(size_t i = 0, end = m_grads.size(); i != end; ++i)
		{
			// update mean
			m_mean(i) = m_beta1 * m_mean(i) + (1 - m_beta1) * m_grads(i);
			
			// update variance
			m_variance(i) = m_beta2 * m_variance(i) + (1 - m_beta2) * m_grads(i) * m_grads(i);
			
			// update parameters
			m_parameters(i) -= lr * m_mean(i) / (sqrt(m_variance(i)) + 1e-8);
		}
		
		return *this;
	}
	
private:
	Tensor &m_parameters;
	Tensor &m_grads;
	Tensor m_mean;
	Tensor m_variance;
	real_t m_learningRate;
	real_t m_beta1;
	real_t m_beta2;
	real_t m_normalize1;
	real_t m_normalize2;
	size_t m_steps;
};

}

#endif
