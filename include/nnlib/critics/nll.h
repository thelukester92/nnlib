#ifndef NLL_H
#define NLL_H

#include "critic.h"

namespace nnlib
{

/// Negative log loss critic.
template <typename T = double>
class NLL : public Critic<T>
{
public:
	NLL(const Storage<size_t> &shape, bool average = true) :
		m_inGrad(shape, true),
		m_average(average)
	{
		NNHardAssertEquals(shape.size(), 2, "Expected matrix input!");
	}
	
	bool average() const
	{
		return m_average;
	}
	
	NLL &average(bool ave)
	{
		m_average = ave;
		return *this;
	}
	
	/// L = 1/n sum_i( -input(target(i)) )
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssertEquals(input.size(0), target.size(0), "Incompatible operands!");
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		NNAssertEquals(target.size(1), 1, "Expected single-column target!");
		
		T sum = 0;
		size_t j;
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			NNAssertGreaterThanOrEquals(target(i, 0), 0, "Expected positive target!");
			j = target(i, 0);
			sum -= input(i, j);
		}
		
		if(m_average)
			sum /= input.size();
		
		return sum;
	}
	
	/// dL/di = target == i ? -1/n : 0
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssertEquals(input.size(0), target.size(0), "Incompatible operands!");
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		NNAssertEquals(target.size(1), 1, "Expected single-column target!");
		
		m_inGrad.fill(0);
		T weight = -1.0;
		
		if(m_average)
			weight /= input.size();
		
		size_t j;
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			NNAssertGreaterThanOrEquals(target(i, 0), 0, "Expected positive target!");
			j = target(i, 0);
			m_inGrad(i, j) = weight;
		}
		
		return m_inGrad;
	}
	
	/// Input gradient buffer.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}

private:
	Tensor<T> m_inGrad;	///< The gradient of the loss w.r.t. the input.
	bool m_average;		///< Whether to average the result. Default: true.
};

}

#endif
