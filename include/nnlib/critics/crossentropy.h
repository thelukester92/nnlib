#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "critic.h"

namespace nnlib
{

/// Cross entropy critic.
template <typename T = double>
class CrossEntropy : public Critic<double>
{
public:
	template <typename M>
	CrossEntropy(const M &model) :
		m_output(model.outputs(), true),
		m_inGrad(model.outputs(), true)
	{}
	
	CrossEntropy(const Storage<size_t> &shape) :
		m_output(shape, true),
		m_inGrad(shape, true)
	{}
	
	/// Loss = ???
	virtual Tensor<T> &forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		
		return m_output;
	}
	
	/// Grad(Loss) = ???
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		
		return m_inGrad;
	}
	
	/// Output buffer (the loss).
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Input gradient buffer.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}

private:
	Tensor<T> m_output;	///< The loss.
	Tensor<T> m_inGrad;	///< The gradient of the loss w.r.t. the input.
};

}

#endif
