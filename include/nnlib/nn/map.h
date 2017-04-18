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
	/// Single element forward.
	virtual T forward(const T &x) = 0;
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) = 0;
	
	// MARK: Module methods
	
	/// Change the input dimensions of this module.
	virtual void resizeInput(const Storage<size_t> &dims) override
	{
		m_inBlame.resize(dims);
		m_output.resize(dims);
	}
	
	/// Change the output dimensions of this module.
	virtual void resizeOutput(const Storage<size_t> &dims) override
	{
		m_inBlame.resize(dims);
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
	
	/// Backward propagate input and output blame, returning input blame.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outBlame) override
	{
		NNAssert(input.shape() == m_output.shape(), "Incompatible input shape!");
		NNAssert(outBlame.shape() == m_output.shape(), "Incompatible output blame shape!");
		auto i = input.begin(), j = input.end(), b = outBlame.begin();
		for(auto k = m_output.begin(), l = m_inBlame.begin(); i != j; ++i, ++b, ++k, ++l)
		{
			*l = *b * backward(*i, *k);
		}
		return m_inBlame;
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Cached input blame.
	virtual Tensor<T> &inBlame() override
	{
		return m_inBlame;
	}
	
private:
	Tensor<T> m_inBlame;
	Tensor<T> m_output;
};

}

#endif
