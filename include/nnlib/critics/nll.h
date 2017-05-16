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
	NLL(const Storage<size_t> &shape) :
		m_inGrad(shape, true)
	{
		NNAssert(shape.size() == 2, "Input to NLL must be a matrix!");
	}
	
	/// L = 1/n sum_i( -input(target(i)) )
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.size(0) == target.size(0), "Incompatible operands to NLL!");
		NNAssert(input.dims() == 2, "Input to NLL must be a Matrix!");
		NNAssert(target.size(1) == 1, "Target for NLL must be a single integer!");
		
		T sum = 0;
		size_t j;
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			NNAssert(target(i, 0) >= 0, "Target for NLL must be positive!");
			j = target(i, 0);
			sum -= input(i, j);
		}
		
		return sum / input.size(0);
	}
	
	/// dL/di = target == i ? -1 : 0
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.size(0) == target.size(0), "Incompatible operands to NLL!");
		NNAssert(input.dims() == 2, "Input to NLL must be a Matrix!");
		NNAssert(target.size(1) == 1, "Target for NLL must be a single integer!");
		
		m_inGrad.fill(0);
		T weight = -1.0 / input.size(0);
		
		size_t j;
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			NNAssert(target(i, 0) >= 0, "Target for NLL must be positive!");
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
};

}

#endif
