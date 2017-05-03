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
	LSTM(size_t inps, size_t outs, size_t bats = 1) :
		m_inpGateX(new Linear<T>(inps, outs, bats)),
		m_inpGateY(new Linear<T>(outs, outs, bats)),
		m_inpGateH(new Linear<T>(outs, outs, bats)),
		m_inpGate(new Logistic<T>(outs, bats)),
		m_fgtGateX(new Linear<T>(inps, outs, bats)),
		m_fgtGateY(new Linear<T>(outs, outs, bats)),
		m_fgtGateH(new Linear<T>(outs, outs, bats)),
		m_fgtGate(new Logistic<T>(outs, bats)),
		m_inpModX(new Linear<T>(inps, outs, bats)),
		m_inpModY(new Linear<T>(outs, outs, bats)),
		m_inpMod(new TanH<T>(outs, bats)),
		m_outGateX(new Linear<T>(inps, outs, bats)),
		m_outGateY(new Linear<T>(outs, outs, bats)),
		m_outGateH(new Linear<T>(outs, outs, bats)),
		m_outGate(new Logistic<T>(outs, bats)),
		m_outMod(new TanH<T>(outs, bats)),
		m_inGrad(bats, inps),
		m_inpAdd(bats, outs),
		m_fgtAdd(bats, outs),
		m_outAdd(bats, outs),
		m_outGrad(bats, outs),
		m_state(bats, outs),
		m_prevState(bats, outs),
		m_prevOutput(bats, outs),
		m_stateGrad(bats, outs),
		m_gradBuffer(bats, outs),
		m_resetGrad(true)
	{
		Container<T>::add(m_inpGateX);
		Container<T>::add(m_inpGateY);
		Container<T>::add(m_inpGateH);
		Container<T>::add(m_inpGate);
		Container<T>::add(m_fgtGateX);
		Container<T>::add(m_fgtGateY);
		Container<T>::add(m_fgtGateH);
		Container<T>::add(m_fgtGate);
		Container<T>::add(m_inpModX);
		Container<T>::add(m_inpModY);
		Container<T>::add(m_inpMod);
		Container<T>::add(m_outGateX);
		Container<T>::add(m_outGateY);
		Container<T>::add(m_outGateH);
		Container<T>::add(m_outGate);
		Container<T>::add(m_outMod);
		reset();
	}
	
	LSTM &reset()
	{
		m_outMod->output().fill(0);
		m_state.fill(0);
		m_resetGrad = true;
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
		m_prevState.copy(m_state);
		m_prevOutput.copy(m_outMod->output());
		
		// input gate
		m_inpGateX->forward(input);
		m_inpGateX->output().addMM(m_inpGateY->forward(m_prevOutput));
		m_inpGateX->output().addMM(m_inpGateH->forward(m_prevState));
		m_inpGate->forward(m_inpGateX->output());
		
		// forget gate
		m_fgtGateX->forward(input);
		m_fgtGateY->output().addMM(m_fgtGateY->forward(m_prevOutput));
		m_fgtGateY->output().addMM(m_fgtGateH->forward(m_prevState));
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
		return m_outMod->forward(m_outGate->output());
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		m_outGrad.addMM(outGrad);
		
		// backprop to hidden state
		m_stateGrad.copy(m_outGrad).pointwiseProduct(m_outGate->output());
		
		// backprop through output gate (m_outGrad must go last here)
		m_outGate->backward(m_outGateX->output(), m_outGrad);
		m_inGrad.copy(m_outGateX->backward(input, m_outGate->inGrad()));
		m_stateGrad.addMM(m_outGateH->backward(m_state, m_outGate->inGrad()));
		m_outGrad.copy(m_outGateY->backward(m_prevOutput, m_outGate->inGrad()));
		
		// backprop through input value
		m_gradBuffer.copy(m_stateGrad).pointwiseProduct(m_inpGate->output());
		m_inpMod->backward(m_inpModX->output(), m_gradBuffer);
		m_inGrad.addMM(m_inpModX->backward(input, m_inpMod->inGrad()));
		m_outGrad.addMM(m_inpModY->backward(m_prevOutput, m_inpMod->inGrad()));
		
		// backprop through forget gate
		m_gradBuffer.copy(m_stateGrad).pointwiseProduct(m_prevState);
		m_fgtGate->backward(m_fgtGateX->output(), m_gradBuffer);
		m_inGrad.addMM(m_fgtGateX->backward(input, m_fgtGate->inGrad()));
		m_stateGrad.addMM(m_fgtGateH->backward(m_prevState, m_fgtGate->inGrad()));
		m_outGrad.addMM(m_fgtGateY->backward(m_prevOutput, m_fgtGate->inGrad()));
		
		// backprop through input gate
		m_gradBuffer.copy(m_stateGrad).pointwiseProduct(m_inpMod->output());
		m_inpGate->backward(m_inpGateX->output(), m_gradBuffer);
		m_inGrad.addMM(m_inpGateX->backward(input, m_inpGate->inGrad()));
		m_stateGrad.addMM(m_inpGateH->backward(m_prevState, m_inpGate->inGrad()));
		m_outGrad.addMM(m_fgtGateY->backward(m_prevOutput, m_inpGate->inGrad()));
		
		// backprop to hidden state
		m_gradBuffer.copy(m_stateGrad).pointwiseProduct(m_fgtGate->output());
		m_stateGrad.addMM(m_gradBuffer);
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_outMod->output();
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
	/// \todo resizing
private:
	Module<T> *m_inpGateX;
	Module<T> *m_inpGateY;
	Module<T> *m_inpGateH;
	Module<T> *m_inpGate;
	Module<T> *m_fgtGateX;
	Module<T> *m_fgtGateY;
	Module<T> *m_fgtGateH;
	Module<T> *m_fgtGate;
	Module<T> *m_inpModX;
	Module<T> *m_inpModY;
	Module<T> *m_inpMod;
	Module<T> *m_outGateX;
	Module<T> *m_outGateY;
	Module<T> *m_outGateH;
	Module<T> *m_outGate;
	Module<T> *m_outMod;
	
	Tensor<T> m_inGrad;
	Tensor<T> m_inpAdd;
	Tensor<T> m_fgtAdd;
	Tensor<T> m_outAdd;
	Tensor<T> m_outGrad;
	
	Tensor<T> m_state;
	Tensor<T> m_prevState;
	Tensor<T> m_prevOutput;
	Tensor<T> m_stateGrad;
	Tensor<T> m_gradBuffer;
	
	bool m_resetGrad;
};

}

#endif
