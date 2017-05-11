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
		NNAssert(m_critic->inGrad().dims() == 2, "CriticSequencer expects matrix critic as input!");
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
		NNAssert(input.dims() == 3 && target.dims() == 3, "CriticSequencer expects a sequence of batches!");
		NNAssert(input.size() == m_critic->inGrad().size(), "Incompatible input!");
		NNAssert(target.size() == m_critic->inGrad().size(), "Incompatible target!");
		const Storage<size_t> &shape = m_critic->inGrad().shape();
		return m_critic->forward(input.view(shape), target.view(shape));
	}
	
	/// Calculate the gradient of the loss w.r.t. the input.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.dims() == 3 && target.dims() == 3, "CriticSequencer expects a sequence of batches!");
		NNAssert(input.size() == m_critic->inGrad().size(), "Incompatible input!");
		NNAssert(target.size() == m_critic->inGrad().size(), "Incompatible target!");
		const Storage<size_t> &shape = m_critic->inGrad().shape();
		m_critic->backward(input.view(shape), target.view(shape));
		return m_inGrad;
	}
	
	/// Input gradient buffer.
	virtual Tensor<T> &inGrad() override
	{
		return m_critic->inGrad();
	}
	
	/// Set the input shape of this critic, including batch and sequence length.
	virtual CriticSequencer &inputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 3, "CriticSequencer must have 3D input!");
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
