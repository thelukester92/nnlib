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
class LSTM : public Container
{
public:
	using Container::inputs;
	using Container::outputs;
	using Container::batch;
	
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "lstm";
	}
	
	LSTM(size_t inps, size_t outs, size_t bats = 1) :
		m_inpGateX(new Linear(inps, outs, bats)),
		m_inpGateY(new Linear(outs, outs, bats)),
		m_inpGateH(new Linear(outs, outs, bats)),
		m_inpGate(new Logistic(outs, bats)),
		m_fgtGateX(new Linear(inps, outs, bats)),
		m_fgtGateY(new Linear(outs, outs, bats)),
		m_fgtGateH(new Linear(outs, outs, bats)),
		m_fgtGate(new Logistic(outs, bats)),
		m_inpModX(new Linear(inps, outs, bats)),
		m_inpModY(new Linear(outs, outs, bats)),
		m_inpMod(new TanH(outs, bats)),
		m_outGateX(new Linear(inps, outs, bats)),
		m_outGateY(new Linear(outs, outs, bats)),
		m_outGateH(new Linear(outs, outs, bats)),
		m_outGate(new Logistic(outs, bats)),
		m_outMod(new TanH(outs, bats)),
		m_inGrad(bats, inps),
		m_inpAdd(bats, outs),
		m_fgtAdd(bats, outs),
		m_outAdd(bats, outs),
		m_outGrad(bats, outs),
		m_state(bats, outs),
		m_prevState(bats, outs),
		m_prevOutput(bats, outs),
		m_stateGrad(bats, outs),
		m_curStateGrad(bats, outs),
		m_gradBuffer(bats, outs),
		m_resetGrad(true)
	{
		Container::add(m_inpGateX);
		Container::add(m_inpGateY);
		Container::add(m_inpGateH);
		Container::add(m_inpGate);
		Container::add(m_fgtGateX);
		Container::add(m_fgtGateY);
		Container::add(m_fgtGateH);
		Container::add(m_fgtGate);
		Container::add(m_inpModX);
		Container::add(m_inpModY);
		Container::add(m_inpMod);
		Container::add(m_outGateX);
		Container::add(m_outGateY);
		Container::add(m_outGateH);
		Container::add(m_outGate);
		Container::add(m_outMod);
		forget();
	}
	
	LSTM(size_t outs) :
		m_inpGateX(new Linear(0, outs, 1)),
		m_inpGateY(new Linear(outs, outs, 1)),
		m_inpGateH(new Linear(outs, outs, 1)),
		m_inpGate(new Logistic(outs, 1)),
		m_fgtGateX(new Linear(0, outs, 1)),
		m_fgtGateY(new Linear(outs, outs, 1)),
		m_fgtGateH(new Linear(outs, outs, 1)),
		m_fgtGate(new Logistic(outs, 1)),
		m_inpModX(new Linear(0, outs, 1)),
		m_inpModY(new Linear(outs, outs, 1)),
		m_inpMod(new TanH(outs, 1)),
		m_outGateX(new Linear(0, outs, 1)),
		m_outGateY(new Linear(outs, outs, 1)),
		m_outGateH(new Linear(outs, outs, 1)),
		m_outGate(new Logistic(outs, 1)),
		m_outMod(new TanH(outs, 1)),
		m_inGrad(1, 0),
		m_inpAdd(1, outs),
		m_fgtAdd(1, outs),
		m_outAdd(1, outs),
		m_outGrad(1, outs),
		m_state(1, outs),
		m_prevState(1, outs),
		m_prevOutput(1, outs),
		m_stateGrad(1, outs),
		m_curStateGrad(1, outs),
		m_gradBuffer(1, outs),
		m_resetGrad(true)
	{
		Container::add(m_inpGateX);
		Container::add(m_inpGateY);
		Container::add(m_inpGateH);
		Container::add(m_inpGate);
		Container::add(m_fgtGateX);
		Container::add(m_fgtGateY);
		Container::add(m_fgtGateH);
		Container::add(m_fgtGate);
		Container::add(m_inpModX);
		Container::add(m_inpModY);
		Container::add(m_inpMod);
		Container::add(m_outGateX);
		Container::add(m_outGateY);
		Container::add(m_outGateH);
		Container::add(m_outGate);
		Container::add(m_outMod);
		forget();
	}
	
	LSTM &forget()
	{
		m_outMod->output().fill(0);
		m_state.fill(0);
		m_resetGrad = true;
		return *this;
	}
	
	// MARK: Container methods
	
	/// Cannot add a component to this container.
	virtual LSTM &add(Module *component) override
	{
		throw std::runtime_error("Cannot add components to a LSTM module!");
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor &forward(const Tensor &input) override
	{
		m_prevState.copy(m_state);
		m_prevOutput.copy(m_outMod->output());
		
		// input gate
		m_inpGateX->forward(input);
		m_inpGateX->output().addMM(m_inpGateY->forward(m_prevOutput));
		m_inpGateX->output().addMM(m_inpGateH->forward(m_prevState));
		m_inpGate->forward(m_inpGateX->output());
		
		// forget gate
		m_fgtGateX->forward(input);
		m_fgtGateX->output().addMM(m_fgtGateY->forward(m_prevOutput));
		m_fgtGateX->output().addMM(m_fgtGateH->forward(m_prevState));
		m_fgtGate->forward(m_fgtGateX->output());
		
		// input value
		m_inpModX->forward(input);
		m_inpModX->output().addMM(m_inpModY->forward(m_prevOutput));
		m_inpMod->forward(m_inpModX->output());
		
		// update memory cell (hidden state)
		m_inpAdd.copy(m_inpGate->output()).pointwiseProduct(m_inpMod->output());
		m_fgtAdd.copy(m_fgtGate->output()).pointwiseProduct(m_state);
		m_state.copy(m_inpAdd).addMM(m_fgtAdd);
		
		// output gate
		m_outGateX->forward(input);
		m_outGateX->output().addMM(m_outGateY->forward(m_prevOutput));
		m_outGateX->output().addMM(m_outGateH->forward(m_state));
		m_outGate->forward(m_outGateX->output());
		m_outAdd.copy(m_outGate->output()).pointwiseProduct(m_state);
		
		// final output
		return m_outMod->forward(m_outAdd);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor &backward(const Tensor &input, const Tensor &outGrad) override
	{
		if(m_resetGrad)
		{
			m_resetGrad = false;
			m_outGrad.fill(0);
			m_stateGrad.fill(0);
		}
		
		// update output gradient
		m_outGrad.addMM(outGrad);
		m_outMod->backward(m_outAdd, m_outGrad);
		
		// backprop to hidden state
		m_curStateGrad.copy(m_outMod->inGrad()).pointwiseProduct(m_outGate->output());
		m_curStateGrad.addMM(m_stateGrad);
		
		// backprop through output gate
		m_gradBuffer.copy(m_outMod->inGrad()).pointwiseProduct(m_state);
		m_outGate->backward(m_outGateX->output(), m_gradBuffer);
		m_inGrad.copy(m_outGateX->backward(input, m_outGate->inGrad()));
		m_outGrad.copy(m_outGateY->backward(m_prevOutput, m_outGate->inGrad()));
		
		// backprop through input value
		m_gradBuffer.copy(m_curStateGrad).pointwiseProduct(m_inpGate->output());
		m_inpMod->backward(m_inpModX->output(), m_gradBuffer);
		m_inGrad.addMM(m_inpModX->backward(input, m_inpMod->inGrad()));
		m_outGrad.addMM(m_inpModY->backward(m_prevOutput, m_inpMod->inGrad()));
		
		// backprop through forget gate
		m_gradBuffer.copy(m_curStateGrad).pointwiseProduct(m_prevState);
		m_fgtGate->backward(m_fgtGateX->output(), m_gradBuffer);
		m_inGrad.addMM(m_fgtGateX->backward(input, m_fgtGate->inGrad()));
		m_stateGrad.copy(m_fgtGateH->backward(m_prevState, m_fgtGate->inGrad()));
		m_outGrad.addMM(m_fgtGateY->backward(m_prevOutput, m_fgtGate->inGrad()));
		
		// backprop through input gate
		m_gradBuffer.copy(m_curStateGrad).pointwiseProduct(m_inpMod->output());
		m_inpGate->backward(m_inpGateX->output(), m_gradBuffer);
		m_inGrad.addMM(m_inpGateX->backward(input, m_inpGate->inGrad()));
		m_stateGrad.addMM(m_inpGateH->backward(m_prevState, m_inpGate->inGrad()));
		m_outGrad.addMM(m_inpGateY->backward(m_prevOutput, m_inpGate->inGrad()));
		
		// backprop to hidden state
		m_gradBuffer.copy(m_curStateGrad).pointwiseProduct(m_fgtGate->output());
		m_stateGrad.addMM(m_gradBuffer);
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor &output() override
	{
		return m_outMod->output();
	}
	
	/// Cached input gradient.
	virtual Tensor &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including batch.
	virtual LSTM &inputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "LSTM expects matrix inputs!");
		
		m_inpGateX->inputs(dims);
		m_fgtGateX->inputs(dims);
		m_inpModX->inputs(dims);
		m_outGateX->inputs(dims);
		m_inGrad.resize(dims);
		
		return batch(dims[0]);
	}
	
	/// Set the output shape of this module, including batch.
	virtual LSTM &outputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 2, "LSTM expects matrix outputs!");
		
		m_inpGateY->outputs(dims);
		m_inpGateH->outputs(dims);
		m_inpGate->outputs(dims);
		m_fgtGateY->outputs(dims);
		m_fgtGateH->outputs(dims);
		m_fgtGate->outputs(dims);
		m_inpModY->outputs(dims);
		m_inpMod->outputs(dims);
		m_outGateY->outputs(dims);
		m_outGateH->outputs(dims);
		m_outGate->outputs(dims);
		m_outMod->outputs(dims);
		
		m_inpAdd.resize(dims);
		m_fgtAdd.resize(dims);
		m_outAdd.resize(dims);
		m_outGrad.resize(dims);
		m_state.resize(dims);
		m_prevState.resize(dims);
		m_prevOutput.resize(dims);
		m_stateGrad.resize(dims);
		m_curStateGrad.resize(dims);
		m_gradBuffer.resize(dims);
		
		return batch(dims[0]);
	}
	
	/// Set the batch size of this module.
	virtual LSTM &batch(size_t bats) override
	{
		Container::batch(bats);
		
		m_inGrad.resizeDim(0, bats);
		m_inpAdd.resizeDim(0, bats);
		m_fgtAdd.resizeDim(0, bats);
		m_outAdd.resizeDim(0, bats);
		m_outGrad.resizeDim(0, bats);
		m_state.resizeDim(0, bats);
		m_prevState.resizeDim(0, bats);
		m_prevOutput.resizeDim(0, bats);
		m_stateGrad.resizeDim(0, bats);
		m_curStateGrad.resizeDim(0, bats);
		m_gradBuffer.resizeDim(0, bats);
		
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor *> stateList() override
	{
		Storage<Tensor *> states = Container::stateList();
		states.push_back(&m_state);
		states.push_back(&m_prevState);
		states.push_back(&m_prevOutput);
		states.push_back(&m_outAdd);
		return states;
	}
	
private:
	Module *m_inpGateX;
	Module *m_inpGateY;
	Module *m_inpGateH;
	Module *m_inpGate;
	Module *m_fgtGateX;
	Module *m_fgtGateY;
	Module *m_fgtGateH;
	Module *m_fgtGate;
	Module *m_inpModX;
	Module *m_inpModY;
	Module *m_inpMod;
	Module *m_outGateX;
	Module *m_outGateY;
	Module *m_outGateH;
	Module *m_outGate;
	Module *m_outMod;
	
	Tensor m_inGrad;
	Tensor m_inpAdd;
	Tensor m_fgtAdd;
	Tensor m_outAdd;
	Tensor m_outGrad;
	
	Tensor m_state;
	Tensor m_prevState;
	Tensor m_prevOutput;
	Tensor m_stateGrad;
	Tensor m_curStateGrad;
	Tensor m_gradBuffer;
	
	bool m_resetGrad;
};

}

#endif
