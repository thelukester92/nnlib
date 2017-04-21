#ifndef RMSPROP_H
#define RMSPROP_H

#include "optimizer.h"

namespace nnlib
{

template <template <typename> class M, template <typename> class C, typename T = double>
class RMSProp : public Optimizer<M, C, T>
{
using Optimizer<M, C, T>::m_model;
using Optimizer<M, C, T>::m_critic;
public:
	RMSProp(M<T> &model, C<T> &critic) :
		Optimizer<M, C, T>(model, critic),
		m_learningRate(0.001),
		m_gamma(0.9)
	{
		m_parameters = Tensor<T>::flatten(model.parameters());
		m_grads = Tensor<T>::flatten(model.grad());
		m_velocity.resize(m_grads.size()).fill(0.0);
		m_meanSquare.resize(m_grads.size()).fill(0.0);
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
	
	T isqrt(T number)
	{
		float x = number * 0.5f;
		float y = number;
		long i = *(long *) &y;
		i = 0x5f3759df - (i >> 1);
		y = *(float *) &i;
		return y * (1.5f - (x * y * y));
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual void step(const Tensor<T> &input, const Tensor<T> &target) override
	{
		// calculate gradient
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		
		// update moment
		auto g = m_grads.begin();
		for(T &v : m_velocity)
		{
			v *= m_gamma;
			v += (1 - m_gamma) * *g * *g;
			++g;
		}
		
		// update parameters
		
	}
	
private:
	Tensor<T> m_parameters;
	Tensor<T> m_grads;
	Tensor<T> m_velocity;
	Tensor<T> m_meanSquare;
	T m_learningRate;
	T m_gamma;
};

}

#endif
