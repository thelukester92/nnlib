#ifndef OPT_NADAM_HPP
#define OPT_NADAM_HPP

#include "optimizer.hpp"

namespace nnlib
{

template <typename T = double>
class Nadam : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
public:
	Nadam(Module<T> &model, Critic<T> &critic);
	
	void reset();
	
	Nadam &learningRate(T learningRate);
	T learningRate() const;
	
	Nadam &beta1(T beta1);
	T beta1() const;
	
	Nadam &beta2(T beta2);
	T beta2() const;
	
	virtual Nadam &step(const Tensor<T> &input, const Tensor<T> &target) override;
	
private:
	Tensor<T> &m_parameters;
	Tensor<T> &m_grads;
	Tensor<T> m_mean;
	Tensor<T> m_variance;
	T m_learningRate;
	T m_beta1;
	T m_beta2;
	T m_normalize1;
	T m_normalize2;
};

}

#ifdef NN_REAL_T
	extern template class nnlib::Nadam<NN_REAL_T>;
#else
	#include "detail/nadam.tpp"
#endif

#endif
