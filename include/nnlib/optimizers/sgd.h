#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

namespace nnlib
{

template <template <typename> class M, template <typename> class C, typename T = double>
class SGD : public Optimizer<M, C, T>
{
using Optimizer<M, C, T>::m_model;
using Optimizer<M, C, T>::m_critic;
public:
	SGD(M<T> &model, C<T> &critic) :
		Optimizer<M, C, T>(model, critic),
		m_learningRate(0.001)
	{
		m_parameters = Tensor<T>::flatten(model.parameters());
		m_grads = Tensor<T>::flatten(model.grad());
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
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual void step(const Tensor<T> &input, const Tensor<T> &target) override
	{
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		Algebra<T>::axpy(m_grads, m_parameters, m_learningRate);
	}
	
private:
	Tensor<T> m_parameters;
	Tensor<T> m_grads;
	T m_learningRate;
};

}

#endif
