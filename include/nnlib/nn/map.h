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
		m_inGrad(bats, outs),
		m_output(bats, outs)
	{}
	
	/// Single element forward.
	virtual T forward(const T &x) = 0;
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) = 0;
	
	// MARK: Module methods
	
	/// Change the input dimensions of this module.
	virtual void resizeInput(const Storage<size_t> &dims) override
	{
		m_inGrad.resize(dims);
		m_output.resize(dims);
	}
	
	/// Change the output dimensions of this module.
	virtual void resizeOutput(const Storage<size_t> &dims) override
	{
		m_inGrad.resize(dims);
		m_output.resize(dims);
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
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
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
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
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
private:
	Tensor<T> m_inGrad;
	Tensor<T> m_output;
};

}

#endif
