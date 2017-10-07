#ifndef CRITICS_CRITIC_HPP
#define CRITICS_CRITIC_HPP

#include "../core/tensor.hpp"

namespace nnlib
{

template <typename T = double>
class Critic
{
public:
	virtual ~Critic() {}
	
	/// Calculate the loss (how far input is from target).
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	/// Calculate the gradient of the loss w.r.t. the input.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	/// Get cached input gradient.
	Tensor<T> &inGrad()
	{
		return m_inGrad;
	}
	
	/// Get cached input gradient.
	const Tensor<T> &inGrad() const
	{
		return m_inGrad;
	}
	
protected:
	Tensor<T> m_inGrad;
};

}

#endif
