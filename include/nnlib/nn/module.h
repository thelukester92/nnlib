#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "../util/tensor.h"

namespace nnlib
{

/// The abtract base class for all neural network modules.
class Module
{
public:
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "module";
	}
	
	virtual ~Module() {}
	
	/// Forward propagate input, returning output.
	virtual Tensor &forward(const Tensor &input) = 0;
	
	/// Forward propagate input, returning output.
	/// Automatically resize to fit, if possible, without changing weights.
	virtual Tensor &safeForward(const Tensor &input)
	{
		safeInputs(input.shape());
		return forward(input);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor &backward(const Tensor &input, const Tensor &outGrad) = 0;
	
	/// Backward propagate input and output gradient, returning input gradient.
	/// Automatically resize to fit, if possible, without changing weights.
	virtual Tensor &safeBackward(const Tensor &input, const Tensor &outGrad)
	{
		safeInputs(input.shape());
		safeOutputs(outGrad.shape());
		return backward(input, outGrad);
	}
	
	/// Cached output.
	virtual Tensor &output() = 0;
	
	/// Cached input gradient.
	virtual Tensor &inGrad() = 0;
	
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
		return const_cast<Module *>(this)->inGrad().shape();
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
		return const_cast<Module *>(this)->output().shape();
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
		return const_cast<Module *>(this)->inGrad().size(0);
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
	virtual Storage<Tensor *> parameterList()
	{
		return {};
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' gradient.
	virtual Storage<Tensor *> gradList()
	{
		return {};
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	/// By default, this is only the calculated output.
	virtual Storage<Tensor *> stateList()
	{
		return { &output() };
	}
	
	/// A flattened tensor of all of this module's parameters.
	/// \todo Don't recalculate each time; can this be cached?
	Tensor &parameters()
	{
		return m_flatParameters = Tensor::flatten(parameterList());
	}
	
	/// A flattened tensor of all of this module's parameters' gradients.
	/// \todo Don't recalculate each time; can this be cached?
	Tensor &grad()
	{
		return m_flatGrad = Tensor::flatten(gradList());
	}
	
	/// A flattened tensor of all of this module's internal states.
	/// \todo Don't recalculate each time; can this be cached?
	Tensor &state()
	{
		return m_flatState = Tensor::flatten(stateList());
	}
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// By default, modules are not serializable.
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const
	{
		throw std::runtime_error("This type is not serializable!");
	}
	
	/// \brief Read from an archive.
	///
	/// By default, modules are not serializable.
	/// \param in The archive from which to read.
	virtual void load(Archive &in)
	{
		throw std::runtime_error("This type is not serializable!");
	}
	
protected:
	Tensor m_flatParameters;
	Tensor m_flatGrad;
	Tensor m_flatState;
};

}

#endif
