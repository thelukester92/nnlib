#ifndef CRITIC_SEQUENCER_H
#define CRITIC_SEQUENCER_H

#include "critic.h"

namespace nnlib
{

template <typename T = double>
class CriticSequencer : public Critic<T>
{
public:
	using Critic<T>::inputs;
	
	CriticSequencer(Critic<T> *critic, size_t sequenceLength = 1) :
		m_critic(critic)
	{
		NNHardAssertEquals(m_critic->inGrad().dims(), 2, "Expected matrix input!");
		Storage<size_t> shape = { sequenceLength };
		for(const size_t &d : m_critic->inputs())
			shape.push_back(d);
		m_inGrad.resize(shape);
		m_critic->inGrad() = m_inGrad.view(shape[0] * shape[1], shape[2]);
	}
	
	virtual ~CriticSequencer()
	{
		delete m_critic;
	}
	
	/// Get the inner critic.
	Critic<T> &critic()
	{
		return *m_critic;
	}
	
	/// Set the length of the sequence this critic uses.
	CriticSequencer &sequenceLength(size_t sequenceLength)
	{
		m_inGrad.resizeDim(0, sequenceLength);
		m_critic->inGrad().resizeDim(0, m_inGrad.shape()[0] * m_inGrad.shape()[1]);
		return *this;
	}
	
	/// Get the length of the sequence this critic uses.
	size_t sequenceLength() const
	{
		return m_inGrad.size(0);
	}
	
	/// Calculate the loss (how far input is from target).
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssertEquals(input.dims(), 3, "Expected a sequence of batches!");
		NNAssertEquals(target.dims(), 3, "Expected a sequence of batches!");
		return m_critic->forward(
			input.view(input.size(0) * input.size(1), input.size(2)),
			target.view(target.size(0) * target.size(1), target.size(2))
		);
	}
	
	/// Calculate the gradient of the loss w.r.t. the input.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssertEquals(input.dims(), 3, "Expected a sequence of batches!");
		NNAssertEquals(target.dims(), 3, "Expected a sequence of batches!");
		m_critic->backward(
			input.view(input.size(0) * input.size(1), input.size(2)),
			target.view(target.size(0) * target.size(1), target.size(2))
		);
		return m_inGrad;
	}
	
	/// Input gradient buffer.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this critic, including batch and sequence length.
	virtual CriticSequencer &inputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 3, "Expected 3D input!");
		m_inGrad.resize(dims);
		m_critic->inGrad() = m_inGrad.view(dims[0] * dims[1], dims[2]);
		return *this;
	}
	
	/// Set the batch size of this critic.
	virtual CriticSequencer &batch(size_t bats) override
	{
		m_inGrad.resizeDim(1, bats);
		m_critic->inGrad().resizeDim(0, m_inGrad.size(0) * m_inGrad.size(1));
		return *this;
	}
	
	/// Get the batch size of this critic.
	virtual size_t batch() const override
	{
		return m_inGrad.size(1);
	}
	
private:
	Critic<T> *m_critic;
	Tensor<T> m_inGrad;
};

}

#endif
