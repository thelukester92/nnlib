#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../util/tensor.h"

namespace nnlib
{

class Module;
class Critic;

class Optimizer
{
public:
	Optimizer(Module &model, Critic &critic) :
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
	Module &model()
	{
		return m_model;
	}
	
	/// Get the critic.
	Critic &critic()
	{
		return m_critic;
	}
	
	/// Perform a single step of training given an input and a target.
	virtual Optimizer &step(const Tensor &input, const Tensor &target) = 0;
	
	/// Perform a single step of training given an input and a target.
	/// Safely (without ruining weights) resize if possible.
	virtual Optimizer &safeStep(const Tensor &input, const Tensor &target)
	{
		m_model.safeResize(input.shape(), target.shape());
		m_critic.inputs(m_model.outputs());
		return step(input, target);
	}
	
protected:
	Module &m_model;
	Critic &m_critic;
};

}

#endif
