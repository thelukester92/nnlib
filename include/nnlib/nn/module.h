#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "../util/tensor.h"

namespace nnlib
{

/// The abtract base class for all neural network modules.
template <typename T = double>
class Module
{
public:
	virtual ~Module() {}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) = 0;
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) = 0;
	
	/// Cached output.
	virtual Tensor<T> &output() = 0;
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() = 0;
	
	/// Get the input shape of this module, including batch.
	virtual const Storage<size_t> &inputs() const
	{
		return const_cast<Module<T> *>(this)->inGrad().shape();
	}
	
	/// Set the input shape of this module, including batch.
	/// By default, this resizes the input gradient and resets the batch to dims[0].
	virtual Module &inputs(const Storage<size_t> &dims)
	{
		inGrad().resize(dims);
		return batch(dims[0]);
	}
	
	/// Get the output shape of this module, including batch.
	virtual const Storage<size_t> &outputs() const
	{
		return const_cast<Module<T> *>(this)->output().shape();
	}
	
	/// Set the output shape of this module, including batch.
	/// By default, this resizes the output and resets the batch to dims[0].
	virtual Module &outputs(const Storage<size_t> &dims)
	{
		output().resize(dims);
		return batch(dims[0]);
	}
	
	/// Get the batch size of this module.
	/// By default, this returns the first dimension of the input shape.
	virtual size_t batch() const
	{
		return const_cast<Module<T> *>(this)->inGrad().size(0);
	}
	
	/// Set the batch size of this module.
	/// By default, this resizes the first dimension of the input gradient and output.
	virtual Module &batch(size_t bats)
	{
		inGrad().resizeDim(0, bats);
		output().resizeDim(0, bats);
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Storage<Tensor<T> *> parameterList()
	{
		return {};
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList()
	{
		return {};
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	/// By default, this is only the calculated output.
	virtual Storage<Tensor<T> *> stateList()
	{
		return { &output() };
	}
	
	/// A flattened tensor of all of this module's parameters.
	Tensor<T> &parameters()
	{
		return m_flatParameters = Tensor<T>::flatten(parameterList());
	}
	
	/// A flattened tensor of all of this module's parameters' gradients.
	Tensor<T> &grad()
	{
		return m_flatGrad = Tensor<T>::flatten(gradList());
	}
	
	/// A flattened tensor of all of this module's internal states.
	Tensor<T> &state()
	{
		return m_flatState = Tensor<T>::flatten(stateList());
	}
	
protected:
	Tensor<T> m_flatParameters;
	Tensor<T> m_flatGrad;
	Tensor<T> m_flatState;
};

}

#endif
