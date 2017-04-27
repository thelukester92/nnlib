#ifndef NN_RECURRENT_H
#define NN_RECURRENT_H

#include "container.h"
#include "sequential.h"
#include "linear.h"
#include "tanh.h"

namespace nnlib
{

/// Vanilla recurrent module.
/// output(t) = squash(feedback(output(t - 1)) + squash(input(t)))
template <typename T = double>
class Recurrent : public Container<T>
{
public:
	Recurrent(size_t inps, size_t outs, size_t bats = 1) :
		m_inputModule(new Linear<T>(inps, outs, bats)),
		m_feedbackModule(new Linear<T>(outs, outs, bats)),
		m_outputModule(new Sequential<T>(new Linear<T>(outs, outs, bats), new TanH<>())),
		m_state(bats, outs),
		m_stateGrad(bats, outs),
		m_prevState(bats, outs)
	{
		this->add(m_outputModule, m_inputModule, m_feedbackModule);
		m_state.fill(0);
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_prevState.copy(m_state);
		m_inputModule->forward(input);
		m_feedbackModule->forward(m_prevState);
		m_state.copy(m_inputModule->output()).addMM(m_feedbackModule->output());
		return m_outputModule->forward(m_state);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		m_outputModule->backward(m_state, outGrad);
		m_outputModule->inGrad().addMM(m_stateGrad);
		m_stateGrad.copy(m_feedbackModule->backward(m_prevState, m_outputModule->inGrad()));
		return m_inputModule->backward(input, m_outputModule->inGrad());
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_outputModule->output();
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inputModule->inGrad();
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' gradient.
	virtual Storage<Tensor<T> *> grad() override
	{
		Storage<Tensor<T> *> grads = Container<T>::grad();
		grads.push_back(&m_stateGrad);
		return grads;
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor<T> *> innerState() override
	{
		Storage<Tensor<T> *> states = Container<T>::innerState();
		states.push_back(&m_state);
		states.push_back(&m_prevState);
		return states;
	}
private:
	Module<T> *m_inputModule;
	Module<T> *m_feedbackModule;
	Module<T> *m_outputModule;
	
	Tensor<T> m_state;
	Tensor<T> m_stateGrad;
	Tensor<T> m_prevState;
};

}

#endif
