#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "module.h"

namespace nnlib
{

/// Base class for optimization.
template <typename M, typename C>
class Optimizer
{
public:
	typedef typename M::type T;
	
	Optimizer(M &model, C &critic) : m_model(model), m_critic(critic) {}
	virtual void optimize(const Matrix<T> &inputs, const Matrix<T> &targets) = 0;
	
protected:
	M &m_model;
	C &m_critic;
};

template <template<typename, typename> class T, typename M, typename C>
T<M, C> MakeOptimizer(M &model, C &critic)
{
	return T<M, C>(model, critic);
}

}

#endif
