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
	CriticSequencer(Critic<T> *critic);
	virtual ~CriticSequencer();
	
	Critic<T> &critic();
	CriticSequencer &critic(Critic<T> *critic);
	
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override;
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override;
	
protected:
	using Critic<T>::m_inGrad;
	
private:
	Critic<T> *m_critic;
};

}

#ifdef NN_REAL_T
	extern template class nnlib::CriticSequencer<NN_REAL_T>;
#else
	#include "detail/criticsequencer.tpp"
#endif

#endif
