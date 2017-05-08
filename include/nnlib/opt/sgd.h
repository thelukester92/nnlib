#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

namespace nnlib
{

template <typename T = double>
class SGD : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
public:
	SGD(Module<T> &model, Critic<T> &critic) :
		Optimizer<T>(model, critic),
		m_learningRate(0.001),
		m_momentum(0)
	{
		m_parameters = Tensor<T>::flatten(model.parameters());
		m_grads = Tensor<T>::flatten(model.grad());
		m_velocity.resize(m_grads.size()).fill(0.0);
	}
	
	SGD &learningRate(T learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	T learningRate() const
	{
		return m_learningRate;
	}
	
	SGD &momentum(T momentum)
	{
		m_momentum = momentum;
		return *this;
	}
	
	T momentum() const
	{
		return m_momentum;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual SGD &step(const Tensor<T> &input, const Tensor<T> &target) override
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
	Tensor<T> m_parameters;
	Tensor<T> m_grads;
	Tensor<T> m_velocity;
	T m_learningRate;
	T m_momentum;
};

}

#endif
