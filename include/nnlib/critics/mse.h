#ifndef MSE_H
#define MSE_H

#include "critic.h"

namespace nnlib
{

/// Mean squared error critic.
/// Assumes the last dimension is inputs, leaving earlier
/// dimensions for batch size and sequence length.
class MSE : public Critic
{
public:
	MSE(const Storage<size_t> &shape, bool average = true) :
		m_inGrad(shape, true),
		m_average(average)
	{
		NNAssert(shape.size() == 2, "Expected matrix input to MSE!");
	}
	
	/// L = 1/n sum_i( (input(i) - target(i))^2 )
	virtual real_t forward(const Tensor &input, const Tensor &target) override
	{
		NNAssert(input.shape() == target.shape(), "Incompatible operands to MSE!");
		NNAssert(input.dims() == 2, "Expected matrix input to MSE!");
		auto tar = target.begin();
		real_t diff, sum = 0;
		for(const real_t &inp : input)
		{
			diff = inp - *tar;
			sum += diff * diff;
			++tar;
		}
		
		if(m_average)
			sum /= input.shape().back();
		
		return sum;
	}
	
	/// dL/di = 2/n (input(i) - target(i))
	virtual Tensor &backward(const Tensor &input, const Tensor &target) override
	{
		NNAssert(input.shape() == target.shape() && input.shape() == m_inGrad.shape(), "Incompatible operands to MSE!");
		NNAssert(input.dims() == 2, "Expected matrix input to MSE!");
		
		real_t norm = 2.0;
		if(m_average)
			norm /= input.shape().back();
		
		auto inp = input.begin(), tar = target.begin();
		for(real_t &g : m_inGrad)
		{
			g = norm * (*inp - *tar);
			++inp;
			++tar;
		}
		
		return m_inGrad;
	}
	
	/// Input gradient buffer.
	virtual Tensor &inGrad() override
	{
		return m_inGrad;
	}

private:
	Tensor m_inGrad;	///< The gradient of the loss w.r.t. the input.
	bool m_average;		///< Whether to average the result. Default: true.
};

}

#endif
