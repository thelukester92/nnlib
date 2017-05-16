#ifndef MSE_H
#define MSE_H

#include "critic.h"

namespace nnlib
{

/// Mean squared error critic.
/// Assumes the last dimension is inputs, leaving earlier
/// dimensions for batch size and sequence length.
template <typename T = double>
class MSE : public Critic<T>
{
public:
	MSE(const Storage<size_t> &shape, bool average = true) :
		m_inGrad(shape, true),
		m_average(average)
	{
		NNAssert(shape.size() == 2, "Expected matrix input to MSE!");
	}
	
	bool average() const
	{
		return m_average;
	}
	
	MSE &average(bool ave)
	{
		m_average = ave;
		return *this;
	}
	
	/// L = 1/n sum_i( (input(i) - target(i))^2 )
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.shape() == target.shape(), "Incompatible operands to MSE!");
		NNAssert(input.dims() == 2, "Expected matrix input to MSE!");
		auto tar = target.begin();
		T diff, sum = 0;
		for(const T &inp : input)
		{
			diff = inp - *tar;
			sum += diff * diff;
			++tar;
		}
		
		if(m_average)
			sum /= input.size();
		
		return sum;
	}
	
	/// dL/di = 2/n (input(i) - target(i))
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.shape() == target.shape() && input.shape() == m_inGrad.shape(), "Incompatible operands to MSE!");
		NNAssert(input.dims() == 2, "Expected matrix input to MSE!");
		
		T norm = 2.0;
		if(m_average)
			norm /= input.size();
		
		auto inp = input.begin(), tar = target.begin();
		for(T &g : m_inGrad)
		{
			g = norm * (*inp - *tar);
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
	Tensor<T> m_inGrad;	///< The gradient of the loss w.r.t. the input.
	bool m_average;		///< Whether to average the result. Default: true.
};

}

#endif
