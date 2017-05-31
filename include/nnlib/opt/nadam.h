#ifndef NADAM_H
#define NADAM_H

#include "optimizer.h"

namespace nnlib
{

template <typename T = double>
class Nadam : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
public:
	Nadam(Module<T> &model, Critic<T> &critic) :
		Optimizer<T>(model, critic),
		m_parameters(model.parameters()),
		m_grads(model.grad()),
		m_learningRate(0.001),
		m_beta1(0.9),
		m_beta2(0.999),
		m_normalize1(1),
		m_normalize2(1)
	{
		m_mean.resize(m_grads.size()).fill(0.0);
		m_variance.resize(m_grads.size()).fill(0.0);
	}
	
	void reset()
	{
		m_normalize1 = 1;
		m_normalize2 = 1;
	}
	
	Nadam &learningRate(T learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	T learningRate() const
	{
		return m_learningRate;
	}
	
	Nadam &beta1(T beta1)
	{
		m_beta1 = beta1;
		return *this;
	}
	
	T beta1() const
	{
		return m_beta1;
	}
	
	Nadam &beta2(T beta2)
	{
		m_beta2 = beta2;
		return *this;
	}
	
	T beta2() const
	{
		return m_beta2;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual Nadam &step(const Tensor<T> &input, const Tensor<T> &target) override
	{
		m_normalize1 *= m_beta1;
		m_normalize2 *= m_beta2;
		
		T lr = m_learningRate / (1 - m_normalize1) * sqrt(1 - m_normalize2);
		
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
			m_parameters(i) -= lr * ((1 - m_beta1) * m_grads(i) + m_beta1 * m_mean(i)) / (sqrt(m_variance(i)) + 1e-8);
		}
		
		return *this;
	}
	
private:
	Tensor<T> &m_parameters;
	Tensor<T> &m_grads;
	Tensor<T> m_mean;
	Tensor<T> m_variance;
	T m_learningRate;
	T m_beta1;
	T m_beta2;
	T m_normalize1;
	T m_normalize2;
};

}

#endif
