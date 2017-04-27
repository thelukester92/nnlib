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
	using Container<T>::add;
	using Container<T>::inputs;
	using Container<T>::outputs;
	using Container<T>::batch;
	
	Recurrent(size_t inps, size_t outs, size_t bats = 1) :
		m_inputModule(new Linear<T>(inps, outs, bats)),
		m_feedbackModule(new Linear<T>(outs, outs, bats)),
		m_outputModule(new Sequential<T>(new Linear<T>(outs, outs, bats), new TanH<>())),
		m_state(bats, outs),
		m_stateGrad(bats, outs),
		m_prevState(bats, outs),
		m_resetStateGrad(true)
	{
		add(m_inputModule, m_feedbackModule, m_outputModule);
		m_state.fill(0);
	}
	
	Recurrent(Module<T> *inputModule, Module<T> *feedbackModule, Module<T> *outputModule) :
		m_inputModule(inputModule),
		m_feedbackModule(feedbackModule),
		m_outputModule(outputModule),
		m_state(m_outputModule->outputs(), true),
		m_stateGrad(m_outputModule->outputs(), true),
		m_prevState(m_outputModule->outputs(), true),
		m_resetStateGrad(true)
	{
		add(m_inputModule, m_feedbackModule, m_outputModule);
		m_state.fill(0);
	}
	
	Recurrent &reset()
	{
		m_state.fill(0);
		return *this;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_prevState.copy(m_state);
		m_inputModule->forward(input);
		m_feedbackModule->forward(m_prevState);
		m_state.copy(m_inputModule->output()).addMM(m_feedbackModule->output());
		m_resetStateGrad = true;
		return m_outputModule->forward(m_state);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		if(m_resetStateGrad)
		{
			m_resetStateGrad = false;
			m_stateGrad.fill(0);
		}
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
	
	/// Set the input shape of this module, including batch.
	virtual Recurrent &inputs(const Storage<size_t> &dims) override
	{
		m_inputModule->inputs(dims);
		return batch(dims[0]);
	}
	
	/// Set the output shape of this module, including batch.
	virtual Recurrent &outputs(const Storage<size_t> &dims) override
	{
		m_feedbackModule->outputs(dims);
		m_outputModule->outputs(dims);
		return batch(dims[0]);
	}
	
	/// Set the batch size of this module.
	virtual Recurrent &batch(size_t bats) override
	{
		Container<T>::batch(bats);
		m_state.resizeDim(0, bats);
		m_stateGrad.resizeDim(0, bats);
		m_prevState.resizeDim(0, bats);
		return *this;
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
	
	bool m_resetStateGrad;
};

}

#endif
