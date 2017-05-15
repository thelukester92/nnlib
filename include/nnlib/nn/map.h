#ifndef NN_MAP_H
#define NN_MAP_H

#include "module.h"

namespace nnlib
{

/// Abstract base class for pointwise functions on inputs, also known as activation functions.
class Map : public Module
{
public:
	using Module::inputs;
	using Module::outputs;
	
	Map(size_t outs = 0, size_t bats = 1) :
		m_inGrad(bats, outs),
		m_output(bats, outs)
	{}
	
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "map";
	}
	
	/// Single element forward.
	virtual T forward(const T &x) = 0;
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) = 0;
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor &forward(const Tensor &input) override
	{
		NNAssert(input.shape() == m_output.shape(), "Incompatible input shape!");
		auto i = input.begin(), j = input.end();
		for(auto k = m_output.begin(); i != j; ++i, ++k)
		{
			*k = forward(*i);
		}
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor &backward(const Tensor &input, const Tensor &outGrad) override
	{
		NNAssert(input.shape() == m_output.shape(), "Incompatible input shape!");
		NNAssert(outGrad.shape() == m_output.shape(), "Incompatible output gradient shape!");
		auto i = input.begin(), j = input.end(), b = outGrad.begin();
		for(auto k = m_output.begin(), l = m_inGrad.begin(); i != j; ++i, ++b, ++k, ++l)
		{
			*l = *b * backward(*i, *k);
		}
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including batch.
	/// In a map, input shape is always equal to output shape.
	virtual Map &inputs(const Storage<size_t> &dims) override
	{
		Module::inputs(dims);
		Module::outputs(dims);
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	/// In a map, input shape is always equal to output shape.
	virtual Map &outputs(const Storage<size_t> &dims) override
	{
		Module::inputs(dims);
		Module::outputs(dims);
		return *this;
	}
	
private:
	Tensor m_inGrad;
	Tensor m_output;
};

}

#endif
