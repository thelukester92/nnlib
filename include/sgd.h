#ifndef SGD_H
#define SGD_H

#include "optimizer.h"
#include "vector.h"

namespace nnlib
{

template <typename M, typename C>
class SGD : public Optimizer<M, C>
{
using Optimizer<M, C>::m_model;
using Optimizer<M, C>::m_critic;
public:
	using T = typename Optimizer<M, C>::T;
	
	SGD(M &model, C &critic, double learningRate = 0.001)
	: Optimizer<M, C>(model, critic), m_parameters(model.parameters()), m_blame(model.blame()), m_learningRate(learningRate) {}
	
	virtual void optimize(const Matrix<T> &inputs, const Matrix<T> &targets) override
	{
		m_model.forward(inputs);
		m_critic.calculateBlame(inputs, targets);
		m_model.backward(inputs, m_critic.blame());
		m_parameters.addScaled(m_blame, m_learningRate);
	}
	
private:
	Vector<T> m_parameters, m_blame;
	T m_learningRate;
};

}

#endif
