#ifndef CRITIC_H
#define CRITIC_H

#include "../util/tensor.h"

namespace nnlib
{

template <typename T = double>
class Critic
{
public:
	virtual ~Critic() {}
	
	/// Calculate the loss (how far input is from target).
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	/// Calculate the loss (how far input is from target).
	/// Automatically resize to fit.
	virtual T safeForward(const Tensor<T> &input, const Tensor<T> &target)
	{
		inputs(input.shape());
		return forward(input, target);
	}
	
	/// Calculate the gradient of the loss w.r.t. the input.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	//// Calculate the gradient of the loss w.r.t. the input.
	/// Automatically resize to fit.
	virtual Tensor<T> &safeBackward(const Tensor<T> &input, const Tensor<T> &target)
	{
		inputs(input.shape());
		return backward(input, target);
	}
	
	/// Input gradient buffer.
	virtual Tensor<T> &inGrad() = 0;
	
	/// Get the input shape of this critic, including batch.
	virtual const Storage<size_t> &inputs() const
	{
		return const_cast<Critic<T> *>(this)->inGrad().shape();
	}
	
	/// Set the input shape of this critic, including batch.
	/// By default, this resizes the input gradient and resets the batch to dims[0].
	virtual Critic &inputs(const Storage<size_t> &dims)
	{
		NNAssert(dims.size() == 2, "Critic expects matrix inputs!");
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
