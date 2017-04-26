#ifndef NN_RECURRENT_H
#define NN_RECURRENT_H

#include "container.h"
#include "sequential.h"
#include "linear.h"
#include "tanh.h"

namespace nnlib
{

/// A vanilla recurrent layer.
template <typename T = double>
class Recurrent : public Container<T>
{
public:
	using Container<T>::add;
	using Container<T>::inputs;
	using Container<T>::outputs;
	
	Recurrent(size_t inps, size_t outs, size_t bats = 1) :
		m_inputModule(new Linear<T>(inps, outs)),
		m_feedbackModule(new Linear<T>(outs, outs)),
		m_outputModule(new Sequential<T>(new Linear<T>(outs, outs), new TanH<>())),
		m_inputs(0, bats, outs),
		m_feedbacks(0, bats, outs),
		m_outputs(0, bats, outs),
		m_buffer(bats, outs),
		m_outGrad(bats, outs),
		m_step(0)
	{
		add(m_inputModule, m_feedbackModule, m_outputModule);
	}
	
	Recurrent &reset()
	{
		m_inputModule->reset();
		m_feedbackModule->reset();
		dynamic_cast<Linear<T> *>(m_outputModule->component(0))->reset();
		m_outGrad.fill(0);
		m_step = 0;
		return *this;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		Tensor<T> prevOutput = m_step == 0 ?
			Tensor<T>(m_outputs.size(1), m_outputs.size(2)) :
			m_outputs.select(0, m_step - 1);
		
		m_inputs.resize(m_step + 1, m_inputs.size(1), m_inputs.size(2));
		m_inputs.select(0, m_step).copy(m_inputModule->forward(input));
		
		m_feedbacks.resize(m_step + 1, m_feedbacks.size(1), m_feedbacks.size(2));
		m_feedbacks.select(0, m_step).copy(m_feedbackModule->forward(prevOutput));
		
		m_buffer.zeros().addMM(m_inputs.select(0, m_step)).addMM(m_feedbacks.select(0, m_step));
		
		m_outputs.resize(m_step + 1, m_outputs.size(1), m_outputs.size(2));
		m_outputs.select(0, m_step).copy(m_outputModule->forward(m_buffer));
		++m_step;
		
		return m_outputModule->output();
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	/// \note Recurrent modules are stateful and expect forward to have been run before backward.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssert(m_step > 0, "Cannot go backward any more!");
		--m_step;
		
		Tensor<T> prevOutput = m_step == 0 ?
			Tensor<T>(m_outputs.size(1), m_outputs.size(2)) :
			m_outputs.select(0, m_step - 1);
		
		m_buffer.zeros().addMM(m_inputs.select(0, m_step)).addMM(m_feedbacks.select(0, m_step));
		
		m_outGrad.addMM(outGrad);
		m_outputModule->backward(m_buffer, m_outGrad);
		m_feedbackModule->backward(prevOutput, m_outputModule->inGrad());
		m_inputModule->backward(input, m_outputModule->inGrad());
		m_outGrad.copy(m_feedbackModule->inGrad());
		
		return m_inputModule->inGrad();
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
		NNAssert(dims.size() == 2, "Recurrent only works with matrix inputs!");
		m_inputModule->inputs(dims);
		return reset();
	}
	
	/// Set the output shape of this module, including batch.
	virtual Recurrent &outputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "Recurrent only works with matrix outputs!");
		m_inputModule->outputs(dims);
		m_feedbackModule->outputs(dims);
		m_outputModule->outputs(dims);
		m_inputs.resize(m_inputs.size(0), dims[0], dims[1]);
		m_feedbacks.resize(m_feedbacks.size(0), dims[0], dims[1]);
		m_outputs.resize(m_outputs.size(0), dims[0], dims[1]);
		m_buffer.resize(dims[0], dims[1]);
		m_outGrad.resize(dims[0], dims[1]);
		return reset();
	}
	
private:
	Linear<T> *m_inputModule;		///< Process inputs to be the size of the outputs.
	Linear<T> *m_feedbackModule;	///< Process past output.
	Sequential<T> *m_outputModule;	///< Combine input and feedback for new output.
	Tensor<T> m_inputs;				///< Processed inputs for each time step.
	Tensor<T> m_feedbacks;			///< Feedbacks for each time step.
	Tensor<T> m_outputs;			///< Outputs for each time step.
	Tensor<T> m_buffer;				///< Input + Feedback temporary buffer.
	Tensor<T> m_outGrad;			///< outGrad(t - 1).
	size_t m_step;					///< Current time step.
};

}

#endif
