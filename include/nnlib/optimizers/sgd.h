#ifndef SGD_H
#define SGD_H

#include "../optimizer.h"
#include "../util/vector.h"

namespace nnlib
{

template <typename M, typename C>
class SGD : public Optimizer<M, C>
{
using Optimizer<M, C>::m_model;
using Optimizer<M, C>::m_critic;
public:
	using T = typename Optimizer<M, C>::T;
	
	SGD(M &model, C &critic, T lr = 0.01, T m = 0.0) :
		Optimizer<M, C>(model, critic),
		m_parameters(Vector<T>::flatten(model.parameters())),
		m_blame(Vector<T>::flatten(model.blame())),
		m_learningRate(lr),
		m_momentum(m)
	{}
	
	double learningRate() const
	{
		return m_learningRate;
	}
	
	SGD &learningRate(T lr)
	{
		m_learningRate = lr;
		return *this;
	}
	
	double momentum() const
	{
		return m_momentum;
	}
	
	SGD &momentum(T m)
	{
		m_momentum = m;
		return *this;
	}
	
	virtual void optimize(const Matrix<T> &inputs, const Matrix<T> &targets) override
	{
		m_blame.scale(m_momentum);
		m_model.backward(inputs, m_critic.backward(m_model.forward(inputs), targets));
		m_parameters.addScaled(m_blame, m_learningRate / inputs.rows());
	}
	
private:
	Vector<T> m_parameters, m_blame;
	T m_learningRate, m_momentum;
};

}

#endif
