#ifndef SSE_H
#define SSE_H

#include "critic.h"

namespace nnlib
{

/// Sum squared error critic.
template <typename T = double>
class SSE : public Critic<double>
{
public:
	template <typename M>
	SSE(const M &model) :
		m_output(model.outputs(), true),
		m_inGrad(model.outputs(), true)
	{}
	
	SSE(const Storage<size_t> &shape) :
		m_output(shape, true),
		m_inGrad(shape, true)
	{}
	
	/// L = 1/2 sum_i( (input(i) - target(i))^2 )
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.shape() == target.shape(), "Incompatible operands to SSE!");
		auto tar = target.begin();
		T diff, sum = 0;
		for(const T &inp : input)
		{
			diff = inp - *tar;
			sum += diff * diff;
			++tar;
		}
		return sum * 0.5;
	}
	
	/// dL/di = input(i) - target(i)
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.shape() == target.shape() && input.shape() == m_inGrad.shape(), "Incompatible operands to SSE!");
		auto inp = input.begin(), tar = target.begin();
		for(T &g : m_inGrad)
		{
			g = *inp - *tar;
			++inp;
			++tar;
		}
		return m_inGrad;
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
