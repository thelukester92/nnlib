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
		m_curStateGrad(bats, outs),
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
		forget();
	}
	
	LSTM(size_t outs = 0) :
		m_inpGateX(new Linear<T>(0, outs, 1)),
		m_inpGateY(new Linear<T>(outs, outs, 1)),
		m_inpGateH(new Linear<T>(outs, outs, 1)),
		m_inpGate(new Logistic<T>(outs, 1)),
		m_fgtGateX(new Linear<T>(0, outs, 1)),
		m_fgtGateY(new Linear<T>(outs, outs, 1)),
		m_fgtGateH(new Linear<T>(outs, outs, 1)),
		m_fgtGate(new Logistic<T>(outs, 1)),
		m_inpModX(new Linear<T>(0, outs, 1)),
		m_inpModY(new Linear<T>(outs, outs, 1)),
		m_inpMod(new TanH<T>(outs, 1)),
		m_outGateX(new Linear<T>(0, outs, 1)),
		m_outGateY(new Linear<T>(outs, outs, 1)),
		m_outGateH(new Linear<T>(outs, outs, 1)),
		m_outGate(new Logistic<T>(outs, 1)),
		m_outMod(new TanH<T>(outs, 1)),
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
	virtual LSTM &add(Module<T> *) override
	{
		throw std::runtime_error("Cannot add components to a LSTM module!");
	}
	
	/// Cannot remove a component from this container.
	virtual Module<T> *remove(size_t) override
	{
		throw std::runtime_error("Cannot remove components from a LSTM module!");
	}
	
	/// Cannot remove a component from this container.
	virtual LSTM &clear() override
	{
		throw std::runtime_error("Cannot remove components from a LSTM module!");
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
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
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
	virtual Tensor<T> &output() override
	{
		return m_outMod->output();
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
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
		Container<T>::batch(bats);
		
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
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states = Container<T>::stateList();
		states.push_back(&m_state);
		states.push_back(&m_prevState);
		states.push_back(&m_prevOutput);
		states.push_back(&m_outAdd);
		return states;
	}
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<LSTM>::name
			<< m_inpGateX << m_inpGateY << m_inpGateH << m_inpGate
			<< m_fgtGateX << m_fgtGateY << m_fgtGateH << m_fgtGate
			<< m_inpModX << m_inpModY << m_inpMod
			<< m_outGateX << m_outGateY << m_outGateH << m_outGate
			<< m_outMod;
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in) override
	{
		std::string str;
		in >> str;
		NNAssert(
			str == Binding<LSTM>::name,
			"Unexpected type! Expected '" + Binding<LSTM>::name + "', got '" + str + "'!"
		);
		
		Container<T>::clear();
		
		in >> m_inpGateX >> m_inpGateY >> m_inpGateH >> m_inpGate
			>> m_fgtGateX >> m_fgtGateY >> m_fgtGateH >> m_fgtGate
			>> m_inpModX >> m_inpModY >> m_inpMod
			>> m_outGateX >> m_outGateY >> m_outGateH >> m_outGate
			>> m_outMod;
		
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
		
		size_t bats = m_inpGateX->inputs()[0], inps = m_inpGateX->inputs()[1], outs = m_inpGateX->outputs()[1];
		
		m_inGrad.resize(bats, inps);
		m_inpAdd.resize(bats, outs);
		m_fgtAdd.resize(bats, outs);
		m_outAdd.resize(bats, outs);
		m_outGrad.resize(bats, outs);
		m_state.resize(bats, outs);
		m_prevState.resize(bats, outs);
		m_prevOutput.resize(bats, outs);
		m_stateGrad.resize(bats, outs);
		m_curStateGrad.resize(bats, outs);
		m_gradBuffer.resize(bats, outs);
		m_resetGrad = true;
	}
	
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
	Tensor<T> m_curStateGrad;
	Tensor<T> m_gradBuffer;
	
	bool m_resetGrad;
};

NNSerializable(LSTM<double>, Module<double>);
NNSerializable(LSTM<float>, Module<float>);

}

#endif
