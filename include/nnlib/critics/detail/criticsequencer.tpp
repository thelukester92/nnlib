#ifndef CRITICS_CRITIC_SEQUENCER_TPP
#define CRITICS_CRITIC_SEQUENCER_TPP

#include "../criticsequencer.hpp"

namespace nnlib
{

template <typename T>
CriticSequencer<T>::CriticSequencer(Critic<T> *critic) :
	m_critic(critic)
{}

template <typename T>
CriticSequencer<T>::~CriticSequencer()
{
	delete m_critic;
}

template <typename T>
Critic<T> &CriticSequencer<T>::critic()
{
	return *m_critic;
}

template <typename T>
CriticSequencer<T> &CriticSequencer<T>::critic(Critic<T> *critic)
{
	m_critic = critic;
	return *this;
}

template <typename T>
T CriticSequencer<T>::forward(const Tensor<T> &input, const Tensor<T> &target)
{
	T output = 0;
	for(size_t i = 0, seqLen = input.size(0); i < seqLen; ++i)
		output += m_critic->forward(input.select(0, i), target.select(0, i));
	return output;
}

template <typename T>
Tensor<T> &CriticSequencer<T>::backward(const Tensor<T> &input, const Tensor<T> &target)
{
	m_inGrad.resize(input.shape());
	for(size_t i = 0, seqLen = input.size(0); i < seqLen; ++i)
		m_inGrad.select(0, i).copy(m_critic->backward(input.select(0, i), target.select(0, i)));
	return m_inGrad;
}

}

#endif
