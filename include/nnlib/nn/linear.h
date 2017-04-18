#ifndef NN_LINEAR_H
#define NN_LINEAR_H

#include "module.h"
#include "../util/algebra.h"

namespace nnlib
{

/// A standard feed-forward layer that returns a linear combination of inputs.
template <typename T = double>
class Linear : public Module<T>
{
public:
	/// Standard inps -> outs layer.
	Linear(size_t inps, size_t outs, size_t bats = 1) :
		m_weights(inps, outs),
		m_weightsBlame(inps, outs),
		m_inBlame(bats, inps),
		m_output(bats, outs)
	{
		reset();
	}
	
	/// any -> outs layer; adding to a sequential will set input size.
	Linear(size_t outs) :
		m_weights(0, outs),
		m_weightsBlame(0, outs),
		m_inBlame(1, 0),
		m_output(1, outs)
	{}
	
	/// Set weights to normally distributed random values.
	Linear &reset()
	{
		m_weights.randn();
		return *this;
	}
	
	/// Set the number of inputs.
	Linear &inputs(size_t inps)
	{
		m_inBlame.resize(m_inBlame.size(0), inps);
		return *this;
	}
	
	/// Set the number of outputs.
	Linear &outputs(size_t outs)
	{
		m_output.resize(m_output.size(0), outs);
		return *this;
	}
	
	/// Set the batch size.
	Linear &batch(size_t bats)
	{
		m_inBlame.resize(bats, m_inBlame.size(1));
		m_output.resize(bats, m_output.size(1));
		return *this;
	}
	
	/// Get the weights of this layer.
	Tensor<T> &weights()
	{
		return m_weights;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssert(input.dims() == 2, "Linear expects Matrix input!");
		// output (bats x outs) = input (bats x inps) x weights (inps x outs)
		Algebra<T>::gemm(input, m_weights, m_output);
		return m_output;
	}
	
	/// Backward propagate input and output blame, returning input blame.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outBlame) override
	{
		NNAssert(input.dims() == 2, "Linear expects Matrix input!");
		NNAssert(outBlame.dims() == 2, "Linear expects Matrix output blame!");
		// inBlame (bats x inps) = outBlame (bats x outs) x weights^T (outs x inps)
		Algebra<T>::gemmNT(outBlame, m_weights, m_inBlame);
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
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Storage<Tensor<T> *> parameters() override
	{
		return { &m_weights };
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' blame.
	virtual Storage<Tensor<T> *> blame() override
	{
		return { &m_weightsBlame };
	}
	
private:
	Tensor<T> m_weights;
	Tensor<T> m_weightsBlame;
	
	Tensor<T> m_inBlame;
	Tensor<T> m_output;
};

}

#endif
