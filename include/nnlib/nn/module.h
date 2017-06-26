#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "../tensor.h"

namespace nnlib
{

/// \brief The abtract base class for all neural network modules.
///
/// \note The assignment operator invalidates parameters, grad, and state.
/// They must be reflattened to use them.
template <typename T = double>
class Module
{
public:
	Module() : m_training(true) {}
	Module(const Module &) = delete;
	Module &operator=(const Module &) = delete;
	virtual ~Module() {}
	
	/// Returns whether this module is in training mode.
	virtual bool training() const
	{
		return m_training;
	}
	
	/// Sets whether this module is in training mode.
	virtual Module &training(bool training)
	{
		m_training = training;
		return *this;
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) = 0;
	
	/// Forward propagate input, returning output.
	/// Automatically resize to fit, if possible, without changing weights.
	virtual Tensor<T> &safeForward(const Tensor<T> &input)
	{
		safeInputs(input.shape());
		return forward(input);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) = 0;
	
	/// Backward propagate input and output gradient, returning input gradient.
	/// Automatically resize to fit, if possible, without changing weights.
	virtual Tensor<T> &safeBackward(const Tensor<T> &input, const Tensor<T> &outGrad)
	{
		safeInputs(input.shape());
		safeOutputs(outGrad.shape());
		return backward(input, outGrad);
	}
	
	/// Cached output.
	virtual Tensor<T> &output() = 0;
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() = 0;
	
	/// Set the input and output shapes of this module.
	virtual Module &resize(const Storage<size_t> &inps, const Storage<size_t> &outs)
	{
		inputs(inps);
		return outputs(outs);
	}
	
	/// Safely (never reset weights) set the input and output shapes of this module.
	virtual Module &safeResize(const Storage<size_t> &inps, const Storage<size_t> &outs)
	{
		safeInputs(inps);
		return safeOutputs(outs);
	}
	
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
	
	/// Safely (never reset weights) set the input shape of this module.
	/// By default, this assumes the first dimension (0) is the batch.
	virtual Module &safeInputs(const Storage<size_t> &dims)
	{
		if(inGrad().size(1) == 0)
			inputs(dims);
		else
			batch(dims[0]);
		return *this;
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
	
	/// Safely (never reset weights) set the output shape of this module.
	/// By default, this assumes the first dimension (0) is the batch.
	virtual Module &safeOutputs(const Storage<size_t> &dims)
	{
		if(output().size(1) == 0)
			outputs(dims);
		else
			batch(dims[0]);
		return *this;
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
	
	/// Reset the internal state of this module.
	virtual Module &forget()
	{
		for(Tensor<T> *t : stateList())
			t->fill(0);
		return *this;
	}
	
	/// A flattened tensor of all of this module's parameters.
	/// \todo Don't recalculate each time; can this be cached?
	Tensor<T> &parameters()
	{
		return m_flatParameters = Tensor<T>::flatten(parameterList());
	}
	
	/// A flattened tensor of all of this module's parameters' gradients.
	/// \todo Don't recalculate each time; can this be cached?
	Tensor<T> &grad()
	{
		return m_flatGrad = Tensor<T>::flatten(gradList());
	}
	
	/// A flattened tensor of all of this module's internal states.
	/// \todo Don't recalculate each time; can this be cached?
	Tensor<T> &state()
	{
		return m_flatState = Tensor<T>::flatten(stateList());
	}
	
protected:
	Tensor<T> m_flatParameters;
	Tensor<T> m_flatGrad;
	Tensor<T> m_flatState;
	bool m_training;
};

}

#endif
