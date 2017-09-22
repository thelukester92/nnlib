#ifndef NN_MAP_H
#define NN_MAP_H

#include "module.h"

namespace nnlib
{

/// Abstract base class for pointwise functions on inputs, also known as activation functions.
template <typename T = double>
class Map : public Module<T>
{
public:
	Map(size_t outs = 0, size_t bats = 1) :
		Module<T>({ outs }, { outs })
	{}
	
	Map(const Map &module) :
		Module<T>(module.inputShape(), module.outputShape())
	{}
	
	Map &operator=(const Map &module)
	{
		resizeOutputs(module.m_outputShape);
		resizeInputs(module.m_inputShape);
		return *this;
	}
	
	/// Single element forward.
	virtual T forward(const T &x) = 0;
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) = 0;
	
	// MARK: Serialization
	
	/// Save to a serialized node.
	virtual void save(Serialized &node) const override
	{
		node.set("shape", this->inputShape());
	}
	
	/// Load from a serialized node.
	virtual void load(const Serialized &node) override
	{
		resizeInputs(node.get<Storage<size_t>>("shape"));
	}
	
	// MARK: Computation
	
	/// Forward propagate input, returning output.
	virtual const Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssertEquals(input.shape(), m_output.shape(), "Incompatible input!");
		auto i = input.begin(), j = input.end();
		for(auto k = m_output.begin(); i != j; ++i, ++k)
		{
			*k = forward(*i);
		}
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual const Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.shape(), m_output.shape(), "Incompatible input!");
		NNAssertEquals(outGrad.shape(), m_output.shape(), "Incompatible output!");
		auto i = input.begin(), j = input.end(), b = outGrad.begin();
		for(auto k = m_output.begin(), l = m_inGrad.begin(); i != j; ++i, ++b, ++k, ++l)
		{
			*l = *b * backward(*i, *k);
		}
		return m_inGrad;
	}
	
	// MARK: Size Management
	
	/// Set the output shape of this module.
	/// In a map, input shape is always equal to output shape.
	virtual void resizeOutputs(const Storage<size_t> &dims) override
	{
		Module<T>::resizeInputs(dims);
		Module<T>::resizeOutputs(dims);
	}
	
	/// Set the input shape of this module.
	/// In a map, input shape is always equal to output shape.
	virtual void resizeInputs(const Storage<size_t> &dims) override
	{
		Module<T>::resizeInputs(dims);
		Module<T>::resizeOutputs(dims);
	}
	
	/// Set the input and output shapes of this module.
	/// In a map, input shape is always equal to output shape.
	virtual void resize(const Storage<size_t> &inps, const Storage<size_t> &outs) override
	{
		NNAssertEquals(inps, outs, "Expected input and output sizes to be equal!");
		Module<T>::resizeInputs(inps);
		Module<T>::resizeOutputs(inps);
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
};

}

NNRegisterType(Map, Module);

#endif
