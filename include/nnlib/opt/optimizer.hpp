#ifndef OPT_OPTIMIZER_HPP
#define OPT_OPTIMIZER_HPP

#include "../critics/critic.hpp"
#include "../nn/module.hpp"

namespace nnlib
{

template <typename T = double>
class Optimizer
{
public:
	Optimizer(Module<T> &model, Critic<T> &critic);
	virtual ~Optimizer();
	
	Module<T> &model();
	Critic<T> &critic();
	
	/// Perform a single step of training given an input and a target.
	virtual Optimizer &step(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
protected:
	Module<T> &m_model;
	Critic<T> &m_critic;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::Optimizer<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/optimizer.tpp"
#endif

#endif
