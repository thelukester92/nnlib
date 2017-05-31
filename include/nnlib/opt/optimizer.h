#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../tensor.h"

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
	
	/// Perform a single step of training given an input and a target.
	/// Safely (without ruining weights) resize if possible.
	virtual Optimizer &safeStep(const Tensor<T> &input, const Tensor<T> &target)
	{
		m_model.safeResize(input.shape(), target.shape());
		m_critic.inputs(m_model.outputs());
		return step(input, target);
	}
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const
	{
		throw std::runtime_error("This optimizer does not support serialization!");
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in)
	{
		throw std::runtime_error("This optimizer does not support serialization!");
	}
	
protected:
	Module<T> &m_model;
	Critic<T> &m_critic;
};

}

#endif
