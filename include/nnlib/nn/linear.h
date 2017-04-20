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
		m_bias(outs),
		m_biasBlame(outs),
		m_inBlame(bats, inps),
		m_output(bats, outs),
		m_addBuffer(bats)
	{
		reset();
	}
	
	/// any -> outs layer; adding to a sequential will set input size.
	Linear(size_t outs) :
		m_weights(0, outs),
		m_weightsBlame(0, outs),
		m_bias(outs),
		m_biasBlame(outs),
		m_inBlame(1, 0),
		m_output(1, outs),
		m_addBuffer(1)
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
		m_bias.resize(outs);
		m_biasBlame.resize(outs);
		m_output.resize(m_output.size(0), outs);
		return *this;
	}
	
	/// Set the batch size.
	Linear &batch(size_t bats)
	{
		m_inBlame.resize(bats, m_inBlame.size(1));
		m_output.resize(bats, m_output.size(1));
		m_addBuffer.resize(bats);
		return *this;
	}
	
	/// Get the weights of this module.
	Tensor<T> &weights()
	{
		return m_weights;
	}
	
	/// Get the bias of this module.
	Tensor<T> &bias()
	{
		return m_bias;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssert(input.dims() == 2, "Linear expects Matrix input!");
		
		// output (bats x outs) = input (bats x inps) x weights (inps x outs)
		Algebra<T>::gemm(input, m_weights, m_output);
		
		// output (bats x outs) += addBuffer (bats x 1) x bias (1 x outs)
		Algebra<T>::ger(m_addBuffer, m_bias, m_output);
		
		return m_output;
	}
	
	/// Backward propagate input and output blame, returning input blame.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outBlame) override
	{
		NNAssert(input.dims() == 2, "Linear expects Matrix input!");
		NNAssert(outBlame.dims() == 2, "Linear expects Matrix output blame!");
		
		// biasBlame (outs x 1) += outBlame^T (outs x bats) x addBuffer^T (bats x 1)
		Algebra<T>::gemv(outBlame, m_addBuffer, m_biasBlame, true);
		
		// weightsBlame (inps x outs) += input^T (bats x inps) x outBlame (bats x outs)
		Algebra<T>::gemm(input, outBlame, m_weightsBlame, true);
		
		// inBlame (bats x inps) = outBlame (bats x outs) x weights^T (outs x inps)
		Algebra<T>::gemm(outBlame, m_weights, m_inBlame, false, true);
		
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
		return { &m_weights, &m_bias };
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' blame.
	virtual Storage<Tensor<T> *> blame() override
	{
		return { &m_weightsBlame, &m_biasBlame };
	}
	
private:
	Tensor<T> m_weights;		///< Module weights.
	Tensor<T> m_weightsBlame;	///< Blame on the weights.
	
	Tensor<T> m_bias;			///< Network bias.
	Tensor<T> m_biasBlame;		///< Blame on the bias.
	
	Tensor<T> m_inBlame;		///< Input blame buffer.
	Tensor<T> m_output;			///< Output buffer.
	
	Tensor<T> m_addBuffer;		///< A vector of 1s for outer-producting bias.
};

}

#endif
