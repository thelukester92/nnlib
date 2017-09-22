#ifndef CRITIC_H
#define CRITIC_H

#include "../core/tensor.h"

namespace nnlib
{

template <typename T = double>
class Critic
{
public:
	virtual ~Critic() {}
	
	// MARK: Computation
	
	/// Calculate the loss (how far input is from target).
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	/// Calculate the gradient of the loss w.r.t. the input.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	// MARK: Size Management
	
	/// Set the input shape of this critic, including batch.
	/// By default, this resizes the input gradient and resets the batch to dims[0].
	virtual Critic &inputs(const Storage<size_t> &dims)
	{
		NNAssertEquals(dims.size(), 2, "Expected matrix input!");
		inGrad().resize(dims);
		return batch(dims[0]);
	}
	
	/// Get the batch size of this critic.
	/// By default, this returns the first dimension of the input shape.
	virtual size_t batch() const
	{
		return const_cast<Critic<T> *>(this)->inGrad().size(0);
	}
	
	/// Set the batch size of this critic.
	/// By default, this resizes the first dimension of the input gradient.
	virtual Critic &batch(size_t bats)
	{
		inGrad().resizeDim(0, bats);
		return *this;
	}
};

}

#endif
