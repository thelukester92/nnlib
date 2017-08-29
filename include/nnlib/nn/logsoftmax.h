#ifndef NN_LOG_SOFT_MAX_H
#define NN_LOG_SOFT_MAX_H

#include "module.h"

namespace nnlib
{

/// Log soft max module for classification problems.
template <typename T = double>
class LogSoftMax : public Module<T>
{
public:
	using Module<T>::inputs;
	using Module<T>::outputs;
	
	LogSoftMax(size_t outs = 0, size_t bats = 1) :
		m_inGrad(bats, outs),
		m_output(bats, outs)
	{}
	
	LogSoftMax(const LogSoftMax &module) :
		m_inGrad(module.m_inGrad.copy()),
		m_output(module.m_output.copy())
	{}
	
	LogSoftMax &operator=(const LogSoftMax &module)
	{
		m_inGrad = module.m_inGrad.copy();
		m_output = module.m_output.copy();
		return *this;
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
		
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			T max = input.narrow(0, i).max(), sum = 0;
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				sum += exp(input(i, j) - max);
			sum = max + log(sum);
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				m_output(i, j) = input(i, j) - sum;
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
		NNAssertEquals(outGrad.shape(), m_output.shape(), "Incompatible output!");
		
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			T sum = outGrad.narrow(0, i).sum();
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				m_inGrad(i, j) = outGrad(i, j) - exp(m_output(i, j)) * sum;
		}
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input and output shapes of this module.
	/// In LogSoftMax, input shape is always equal to output shape.
	virtual LogSoftMax &resize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssertEquals(inps, outs, "Expected input and output sizes to be equal!");
		return inputs(outs);
	}
	
	/// Safely (never reset weights) set the input and output shapes of this module.
	/// In LogSoftMax, input shape is always equal to output shape.
	virtual LogSoftMax &safeResize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssertEquals(inps, outs, "Expected input and output sizes to be equal!");
		this->safeInputs(inps);
		return *this;
	}
	
	/// Set the input shape of this module, including batch.
	/// In LogSoftMax, input shape is always equal to output shape.
	virtual LogSoftMax &inputs(const Storage<size_t> &dims) override
	{
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	/// In LogSoftMax, input shape is always equal to output shape.
	virtual LogSoftMax &outputs(const Storage<size_t> &dims) override
	{
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		return *this;
	}
	
	/// Save to a serialized node.
	virtual void save(Serialized &node) const override
	{
		node.set("shape", this->inputs());
	}
	
	/// Load from a serialized node.
	virtual void load(const Serialized &node) override
	{
		inputs(node.get<Storage<size_t>>("shape"));
	}
	
private:
	Tensor<T> m_inGrad;	///< Input gradient buffer.
	Tensor<T> m_output;	///< Output buffer.
};

}

NNRegisterType(LogSoftMax<float>, Module<float>);
NNRegisterType(LogSoftMax<double>, Module<double>);

#endif
