#ifndef SSE_H
#define SSE_H

#include "critic.h"

namespace nnlib
{

/// Sum squared error critic.
template <typename T = double>
class SSE : public Critic<double>
{
public:
	SSE(size_t outs = 0, size_t bats = 1) :
		m_output(bats, outs),
		m_inBlame(bats, outs)
	{}
	
	SSE &outputs(size_t outs)
	{
		m_output.resize(m_output.size(0), outs);
		m_inBlame.resize(m_inBlame.size(0), outs);
	}
	
	SSE &batch(size_t bats)
	{
		m_output.resize(bats, m_output.size(1));
		m_inBlame.resize(bats, m_inBlame.size(1));
	}
	
	/// Loss = sum_i( 1/2 * (target(i) - input(i))^2 )
	virtual Tensor<T> &forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.shape() == target.shape() && input.shape() == m_output.shape(), "Incompatible operands to SSE!");
		auto inp = input.begin(), tar = target.begin();
		T diff;
		for(auto out = m_output.begin(), end = m_output.end(); out != end; ++inp, ++tar, ++out)
		{
			diff = *tar - *inp;
			*out = 0.5 * diff * diff;
		}
		return m_output;
	}
	
	/// Grad(Loss) = sum_i( target(i) - input(i) )
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.shape() == target.shape() && input.shape() == m_output.shape(), "Incompatible operands to SSE!");
		auto inp = input.begin(), tar = target.begin();
		for(auto blame = m_inBlame.begin(), end = m_inBlame.end(); blame != end; ++inp, ++tar, ++blame)
		{
			*blame = *tar - *inp;
		}
		return m_inBlame;
	}
	
	/// Output buffer (the loss).
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Input blame buffer (the gradient).
	virtual Tensor<T> &inBlame() override
	{
		return m_inBlame;
	}

private:
	Tensor<T> m_output;		///< The loss.
	Tensor<T> m_inBlame;	///< The gradient of the loss w.r.t. the input.
};

}

#endif
