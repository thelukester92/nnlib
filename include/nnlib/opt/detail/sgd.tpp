#ifndef OPT_SGD_TPP
#define OPT_SGD_TPP

#include "../sgd.hpp"

namespace nnlib
{

template <typename T>
SGD<T>::SGD(Module<T> &model, Critic<T> &critic) :
	Optimizer<T>(model, critic),
	m_parameters(model.params()),
	m_grads(model.grad()),
	m_velocity(m_grads.size(0)),
	m_learningRate(0.001),
	m_momentum(0)
{
	m_velocity.fill(0.0);
}

template <typename T>
SGD<T> &SGD<T>::learningRate(T learningRate)
{
	m_learningRate = learningRate;
	return *this;
}

template <typename T>
T SGD<T>::learningRate() const
{
	return m_learningRate;
}

template <typename T>
SGD<T> &SGD<T>::momentum(T momentum)
{
	m_momentum = momentum;
	return *this;
}

template <typename T>
T SGD<T>::momentum() const
{
	return m_momentum;
}

template <typename T>
SGD<T> &SGD<T>::step(const Tensor<T> &input, const Tensor<T> &target)
{
	// calculate gradient
	m_grads.fill(0);
	m_model.backward(input, m_critic.backward(m_model.forward(input), target));
	
	if(m_momentum)
	{
		// apply momentum
		m_velocity.scale(m_momentum);
		m_velocity.addV(m_grads);
		
		// Nesterov step
		m_grads.addV(m_velocity, m_momentum);
	}
	
	// update parameters
	m_parameters.addV(m_grads, -m_learningRate);
	
	return *this;
}

}

#endif
