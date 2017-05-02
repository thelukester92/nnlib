#ifndef NN_LSTM_H
#define NN_LSTM_H

#include "concat.h"
#include "container.h"
#include "sequential.h"
#include "linear.h"
#include "logistic.h"
#include "tanh.h"

namespace nnlib
{

/// LSTM recurrent module.
template <typename T = double>
class LSTM : public Container<T>
{
public:
	using Container<T>::inputs;
	using Container<T>::outputs;
	using Container<T>::batch;
	
	LSTM(size_t inps, size_t outs, size_t bats = 1) :
		m_inputSquasher(new Sequential<T>(new Linear<T>(inps + outs, outs, bats), new TanH<>())),
		m_outputSquasher(new TanH<>(outs, bats)),
		m_inputGate(new Sequential<T>(new Linear<T>(inps + 2 * outs, outs, bats), new Logistic<T>())),
		m_forgetGate(new Sequential<T>(new Linear<T>(inps + 2 * outs, outs, bats), new Logistic<T>())),
		m_outputGate(new Sequential<T>(new Linear<T>(inps + 2 * outs, outs, bats), new Logistic<T>())),
		m_state(bats, outs),
		m_stateGrad(bats, outs),
		m_prevState(bats, outs),
		m_outGrad(bats, outs),
		m_prevOutput(bats, outs),
		m_inGrad(bats, inps),
		m_xyhBuffer(bats, inps + 2 * outs),
		m_igxBuffer(bats, outs),
		m_fghBuffer(bats, outs),
		m_xyh2Buffer(bats, inps + 2 * outs),
		m_outBuffer(bats, outs),
		m_gradBuffer(bats, outs),
		m_resetStateGrad(true)
	{
		Container<T>::add(m_inputSquasher);
		Container<T>::add(m_inputGate);
		Container<T>::add(m_forgetGate);
		Container<T>::add(m_outputGate);
		Container<T>::add(m_outputSquasher);
		reset();
	}
	
	LSTM(Module<T> *inputSquasher, Module<T> *outputSquasher) :
		m_inputSquasher(inputSquasher),
		m_outputSquasher(outputSquasher),
		m_inputGate(new Logistic<T>()),
		m_forgetGate(new Logistic<T>()),
		m_outputGate(new Logistic<T>()),
		m_resetStateGrad(true)
	{
		inputs({ m_inputSquasher->inputs()[0], m_inputSquasher->inputs()[1] - m_inputSquasher->outputs()[1] });
		outputs(m_outputSquasher->outputs());
		Container<T>::add(m_inputSquasher);
		Container<T>::add(m_inputGate);
		Container<T>::add(m_forgetGate);
		Container<T>::add(m_outputGate);
		Container<T>::add(m_outputSquasher);
		reset();
	}
	
	LSTM &reset()
	{
		m_state.fill(0);
		m_outputSquasher->output().fill(0);
		m_resetStateGrad = true;
		return *this;
	}
	
	// MARK: Container methods
	
	/// Cannot add a component to this container.
	virtual LSTM &add(Module<T> *component) override
	{
		throw std::runtime_error("Cannot add components to a LSTM module!");
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_prevOutput.copy(m_outputSquasher->output());
		m_prevState.copy(m_state);
		m_xyhBuffer.concat(1, input, m_prevOutput, m_prevState);
		
		size_t inps = m_inputSquasher->inputs()[1] - m_inputSquasher->outputs()[1];
		size_t outs = m_outputGate->outputs()[1];
		
		//
		
		auto i = m_xyhBuffer.begin();
		for(const T &v : input)
		{
			*i = 0;
			++i;
		}
		
		m_inputSquasher->forward(m_xyhBuffer.narrow(1, 0, inps + outs));
		m_inputGate->forward(m_xyhBuffer);
		m_igxBuffer.copy(m_inputSquasher->output()).pointwiseProduct(m_inputGate->output());
		m_state.copy(m_igxBuffer);
		
		m_xyh2Buffer.concat(1, input, m_prevOutput, m_state);
		m_outputGate->forward(m_xyh2Buffer);
		// m_resetStateGrad = true;
		
		/*
		
		m_inputSquasher->forward(m_xyhBuffer.narrow(1, 0, inps + outs));
		
		m_inputGate->forward(m_xyhBuffer);
		m_forgetGate->forward(m_xyhBuffer);
		
		m_igxBuffer.copy(m_inputSquasher->output()).pointwiseProduct(m_inputGate->output());
		m_fghBuffer.copy(m_prevState).pointwiseProduct(m_forgetGate->output());
		m_state.copy(m_igxBuffer).addMM(m_fghBuffer);
		
		m_xyh2Buffer.concat(1, input, m_prevOutput, m_state);
		m_outputGate->forward(m_xyh2Buffer);
		
		m_resetStateGrad = true;
		
		*/
		
		return m_outputSquasher->forward(m_outBuffer.copy(m_state).pointwiseProduct(m_outputGate->output()));
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		if(m_resetStateGrad)
		{
			m_resetStateGrad = false;
			m_stateGrad.fill(0);
			m_outGrad.fill(0);
		}
		
		size_t inps = m_inputSquasher->inputs()[1] - m_inputSquasher->outputs()[1];
		size_t outs = m_outputGate->outputs()[1];
		// m_outGrad.addMM(outGrad);
		reset();
		m_prevState.fill(0);
		m_state.fill(0);
		m_outputSquasher->output().fill(0);
		for(auto *i : this->grad())
		{
			for(auto &v : *i)
			{
				if(fabs(v) > 0)
				{
					std::cout << "V@! " << v << std::endl;
				}
			}
		}
		
		
		
		/*
		
		// derivative of final output
		m_gradBuffer.copy(m_outGrad).pointwiseProduct(m_outputGate->output());
		m_stateGrad.addMM(m_gradBuffer);
		
		// derivative of output gate
		m_gradBuffer.copy(m_outGrad).pointwiseProduct(m_state);
		m_outputGate->backward(m_xyh2Buffer, m_gradBuffer);
		m_inGrad.addMM(m_outputGate->inGrad().narrow(1, 0, inps));
		m_outGrad.addMM(m_outputGate->inGrad().narrow(1, inps, outs));
		m_stateGrad.addMM(m_outputGate->inGrad().narrow(1, inps + outs, outs));
		
		// derivative of input squasher
		m_gradBuffer.copy(m_stateGrad).pointwiseProduct(m_inputGate->output());
		m_inputSquasher->backward(m_xyhBuffer.narrow(1, 0, inps + outs), m_gradBuffer);
		m_inGrad.addMM(m_inputSquasher->inGrad().narrow(1, 0, inps));
		m_outGrad.addMM(m_inputSquasher->inGrad().narrow(1, inps, outs));
		
		// derivative of input gate
		m_gradBuffer.copy(m_stateGrad).pointwiseProduct(m_inputSquasher->output());
		m_inputGate->backward(m_xyhBuffer, m_gradBuffer);
		m_inGrad.addMM(m_inputGate->inGrad().narrow(1, 0, inps));
		m_outGrad.addMM(m_inputGate->inGrad().narrow(1, inps, outs));
		m_stateGrad.addMM(m_inputGate->inGrad().narrow(1, inps + outs, outs));
		
		// derivative of forget gate
		m_gradBuffer.copy(m_stateGrad).pointwiseProduct(m_prevState);
		m_forgetGate->backward(m_xyhBuffer, m_gradBuffer);
		m_inGrad.addMM(m_forgetGate->inGrad().narrow(1, 0, inps));
		m_outGrad.addMM(m_forgetGate->inGrad().narrow(1, inps, outs));
		m_stateGrad.addMM(m_forgetGate->inGrad().narrow(1, inps + outs, outs));
		
		// derivative of state
		m_gradBuffer.copy(m_stateGrad).pointwiseProduct(m_forgetGate->output());
		m_stateGrad.addMM(m_gradBuffer);
		
		*/
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_outputSquasher->output();
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including batch.
	virtual LSTM &inputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "Expected matrix shape for LSTM inputs!");
		size_t bats = dims[0], inps = dims[1], outs = m_outputSquasher->outputs()[1];
		
		m_inputSquasher->inputs(dims);
		m_outputSquasher->batch(bats);
		m_inputGate->inputs({ bats, inps + 2 * outs });
		m_forgetGate->inputs({ bats, inps + 2 * outs });
		m_outputGate->inputs({ bats, 3 * outs });
		
		m_state.resizeDim(0, bats);
		m_stateGrad.resizeDim(0, bats);
		m_prevState.resizeDim(0, bats);
		m_outGrad.resizeDim(0, bats);
		m_prevOutput.resizeDim(0, bats);
		m_inGrad.resize(bats, inps);
		
		m_xyhBuffer.resize(bats, inps + 2 * outs),
		m_igxBuffer.resizeDim(0, bats);
		m_fghBuffer.resizeDim(0, bats);
		m_xyh2Buffer.resize(bats, inps + 2 * outs),
		m_outBuffer.resizeDim(0, bats);
		m_gradBuffer.resizeDim(0, bats);
		
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	virtual LSTM &outputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "Expected matrix shape for LSTM outputs!");
		size_t bats = dims[0], outs = dims[1], inps = m_inputSquasher->inputs()[1];
		
		m_inputSquasher->batch(bats);
		m_outputSquasher->outputs(dims);
		m_inputGate->outputs({ bats, inps + 2 * outs });
		m_forgetGate->outputs({ bats, inps + 2 * outs });
		m_outputGate->outputs({ bats, 3 * outs });
		
		m_state.resize(bats, outs);
		m_stateGrad.resize(bats, outs);
		m_prevState.resize(bats, outs);
		m_outGrad.resize(bats, outs);
		m_prevOutput.resize(bats, outs);
		m_inGrad.resizeDim(0, bats);
		
		m_xyhBuffer.resize(bats, inps + 2 * outs),
		m_igxBuffer.resize(bats, outs),
		m_fghBuffer.resize(bats, outs),
		m_xyh2Buffer.resize(bats, inps + 2 * outs),
		m_outBuffer.resize(bats, outs);
		m_gradBuffer.resize(bats, outs);
		
		return *this;
	}
	
	/// Set the batch size of this module.
	virtual LSTM &batch(size_t bats) override
	{
		Container<T>::batch(bats);
		m_state.resizeDim(0, bats);
		m_stateGrad.resizeDim(0, bats);
		m_prevState.resizeDim(0, bats);
		m_outGrad.resizeDim(0, bats);
		m_prevOutput.resizeDim(0, bats);
		m_inGrad.resizeDim(0, bats);
		m_xyhBuffer.resizeDim(0, bats);
		m_igxBuffer.resizeDim(0, bats);
		m_fghBuffer.resizeDim(0, bats);
		m_xyh2Buffer.resizeDim(0, bats);
		m_outBuffer.resizeDim(0, bats);
		m_gradBuffer.resizeDim(0, bats);
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor<T> *> innerState() override
	{
		Storage<Tensor<T> *> states = Container<T>::innerState();
		states.push_back(&m_state);
		states.push_back(&m_prevState);
		states.push_back(&m_prevOutput);
		states.push_back(&m_xyhBuffer);
		states.push_back(&m_xyh2Buffer);
		return states;
	}
private:
	Module<T> *m_inputSquasher;
	Module<T> *m_outputSquasher;
	Module<T> *m_inputGate;
	Module<T> *m_forgetGate;
	Module<T> *m_outputGate;
	
	Tensor<T> m_state;
	Tensor<T> m_stateGrad;
	Tensor<T> m_prevState;
	Tensor<T> m_outGrad;
	Tensor<T> m_prevOutput;
	
	Tensor<T> m_inGrad;
	
	Tensor<T> m_xyhBuffer;
	Tensor<T> m_igxBuffer;
	Tensor<T> m_fghBuffer;
	Tensor<T> m_xyh2Buffer;
	Tensor<T> m_outBuffer;
	Tensor<T> m_gradBuffer;
	
	bool m_resetStateGrad;
};

}

#endif
