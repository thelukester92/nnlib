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
		m_momentum(0.1),
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
	
	RMSProp &momentum(T momentum)
	{
		m_momentum = momentum;
		return *this;
	}
	
	T momentum() const
	{
		return m_momentum;
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
		// update parameters
		Algebra<T>::axpy(m_velocity, m_parameters, m_momentum);
		
		// calculate gradient
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		
		// put parameters back
		Algebra<T>::axpy(m_velocity, m_parameters, -m_momentum);
		
		// update velocity
		m_velocity.scale(m_momentum);
		Algebra<T>::axpy(m_grads, m_velocity);
		
		// update position
		auto m = m_meanSquare.begin(), v = m_velocity.begin();
		for(auto p = m_parameters.begin(), end = m_parameters.end(); p != end; ++m, ++v, ++p)
		{
			*m = m_gamma * *m + (1 - m_gamma) * *v * *v;
			*p += m_learningRate * *v * isqrt(*m);
		}
	}
	
private:
	Tensor<T> m_parameters;
	Tensor<T> m_grads;
	Tensor<T> m_velocity;
	Tensor<T> m_meanSquare;
	T m_learningRate;
	T m_momentum;
	T m_gamma;
};

}

#endif
