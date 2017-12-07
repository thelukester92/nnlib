#ifndef CRITICS_CRITIC_SEQUENCER_HPP
#define CRITICS_CRITIC_SEQUENCER_HPP

#include "critic.hpp"

namespace nnlib
{

/// \brief Allows an extra "sequence" dimension when passing in error to a critic.
///
/// This not needed for shape-agnostic critics like MSE, but is needed for shape-dependent
/// critics like NLL.
template <typename T = double>
class CriticSequencer : public Critic<T>
{
public:
	CriticSequencer(Critic<T> *critic) :
		m_critic(critic)
	{}
	
	virtual ~CriticSequencer()
	{
		delete m_critic;
	}
	
	/// Get the inner critic.
	Critic<T> &critic()
	{
		return *m_critic;
	}
	
	/// Set the inner critic.
	CriticSequencer &critic(Critic<T> *critic)
	{
		m_critic = critic;
		return *this;
	}
	
	/// MARK: Computation
	
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		T output = 0;
		for(size_t i = 0, seqLen = input.size(0); i < seqLen; ++i)
			output += m_critic->forward(input.select(0, i), target.select(0, i));
		return output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		m_inGrad.resize(input.shape());
		for(size_t i = 0, seqLen = input.size(0); i < seqLen; ++i)
			m_inGrad.select(0, i).copy(m_critic->backward(input.select(0, i), target.select(0, i)));
		return m_inGrad;
	}
	
protected:
	using Critic<T>::m_inGrad;
	
private:
	Critic<T> *m_critic;
};

}

#endif
