#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../util/tensor.h"

namespace nnlib
{

template <template<typename> class M, template<typename> class C, typename T = double>
class Optimizer
{
public:
	Optimizer(M<T> &model, C<T> &critic) :
		m_model(model),
		m_critic(critic)
	{}
	
	/// Get the model.
	M<T> &model()
	{
		return m_model;
	}
	
	/// Get the critic.
	C<T> &critic()
	{
		return m_critic;
	}
	
	/// Perform a single step of training given an input and a target.
	virtual void step(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
protected:
	M<T> &m_model;
	C<T> &m_critic;
};

template <template <template <typename> class, template <typename> class, typename> class O, template <typename> class M, template <typename> class C, typename T = double>
O<M, C, T> makeOptimizer(M<T> &model, C<T> &critic)
{
	return O<M, C, T>(model, critic);
}

}

#endif
