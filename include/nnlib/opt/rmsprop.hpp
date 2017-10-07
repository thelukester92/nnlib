#ifndef OPT_RMSPROP_HPP
#define OPT_RMSPROP_HPP

#include "optimizer.hpp"

namespace nnlib
{

template <typename T = double>
class RMSProp : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
public:
	RMSProp(Module<T> &model, Critic<T> &critic) :
		Optimizer<T>(model, critic),
		m_parameters(model.params()),
		m_grads(model.grad()),
		m_learningRate(0.001),
		m_gamma(0.9)
	{
		m_variance.resize(m_grads.size()).fill(0.0);
	}
	
	RMSProp &learningRate(T learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	T learningRate() const
	{
		return m_learningRate;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual RMSProp &step(const Tensor<T> &input, const Tensor<T> &target) override
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
	Tensor<T> &m_parameters;
	Tensor<T> &m_grads;
	Tensor<T> m_variance;
	T m_learningRate;
	T m_gamma;
};

}

#endif
