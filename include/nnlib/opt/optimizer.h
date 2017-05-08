#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../util/tensor.h"

namespace nnlib
{

template <typename T>
class Module;

template <typename T>
class Critic;

template <typename T = double>
class Optimizer
{
public:
	Optimizer(Module<T> &model, Critic<T> &critic) :
		m_model(model),
		m_critic(critic)
	{}
	
	virtual ~Optimizer() {}
	
	/// Batch the model and critic.
	Optimizer &batch(size_t bats)
	{
		m_model.batch(bats);
		m_critic.batch(bats);
		return *this;
	}
	
	/// Get the model.
	Module<T> &model()
	{
		return m_model;
	}
	
	/// Get the critic.
	Critic<T> &critic()
	{
		return m_critic;
	}
	
	/// Perform a single step of training given an input and a target.
	virtual Optimizer &step(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
protected:
	Module<T> &m_model;
	Critic<T> &m_critic;
};

}

#endif
