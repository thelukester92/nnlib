#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

namespace nnlib
{

class SGD : public Optimizer
{
using Optimizer::m_model;
using Optimizer::m_critic;
public:
	SGD(Module &model, Critic &critic) :
		Optimizer(model, critic),
		m_parameters(model.parameters()),
		m_grads(model.grad()),
		m_velocity(m_grads.size(0)),
		m_learningRate(0.001),
		m_momentum(0)
	{
		m_velocity.fill(0.0);
	}
	
	SGD &learningRate(real_t learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	real_t learningRate() const
	{
		return m_learningRate;
	}
	
	SGD &momentum(real_t momentum)
	{
		m_momentum = momentum;
		return *this;
	}
	
	real_t momentum() const
	{
		return m_momentum;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual SGD &step(const Tensor &input, const Tensor &target) override
	{
		// calculate gradient
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		
		if(m_momentum)
		{
			// apply momentum
			m_velocity.scale(m_momentum);
			m_velocity.addVV(m_grads);
			
			// Nesterov step
			m_grads.addVV(m_velocity, m_momentum);
		}
		
		// update parameters
		m_parameters.addVV(m_grads, -m_learningRate);
		
		return *this;
	}
	
private:
	Tensor &m_parameters;
	Tensor &m_grads;
	Tensor m_velocity;
	real_t m_learningRate;
	real_t m_momentum;
};

}

#endif
