#ifndef NN_RECURRENT_H
#define NN_RECURRENT_H

#include "container.h"
#include "sequential.h"
#include "linear.h"
#include "tanh.h"

namespace nnlib
{

/// A simple recurrent module.
template <typename T = double>
class Recurrent : public Container<T>
{
public:
	using Container<T>::inputs;
	using Container<T>::outputs;
	using Container<T>::batch;
	
	Recurrent(size_t inps, size_t outs, size_t bats = 1) :
		m_inpMod(new Linear<T>(inps, outs, bats)),
		m_memMod(new Linear<T>(outs, outs, bats)),
		m_outMod(new Sequential<T>(new Linear<T>(outs, outs, bats), new TanH<>())),
		m_state(bats, outs),
		m_statePrev(bats, outs),
		m_stateGrad(bats, outs),
		m_resetGrad(true)
	{
		Container<T>::add(m_inpMod);
		Container<T>::add(m_memMod);
		Container<T>::add(m_outMod);
		reset();
	}
	
	Recurrent(Module<T> *inpMod, Module<T> *memMod, Module<T> *outMod) :
		m_inpMod(inpMod),
		m_memMod(memMod),
		m_outMod(outMod),
		m_state(m_outMod->outputs(), true),
		m_statePrev(m_outMod->outputs(), true),
		m_stateGrad(m_outMod->outputs(), true),
		m_resetGrad(true)
	{
		NNAssert(m_inpMod->outputs().size() == 2, "Expected matrix inputs to Recurrent module!");
		NNAssert(m_memMod->outputs().size() == 2, "Expected matrix inputs to Recurrent module!");
		NNAssert(m_outMod->outputs().size() == 2, "Expected matrix inputs to Recurrent module!");
		Container<T>::add(m_inpMod);
		Container<T>::add(m_memMod);
		Container<T>::add(m_outMod);
		reset();
	}
	
	Recurrent &reset()
	{
		m_state.fill(0);
		m_resetGrad = true;
		return *this;
	}
	
	// MARK: Container methods
	
	/// Cannot add a component to this container.
	virtual Recurrent &add(Module<T> *component) override
	{
		throw std::runtime_error("Cannot add components to a recurrent module!");
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_resetGrad = true;
		
		m_statePrev.copy(m_state);
		m_inpMod->forward(input);
		m_memMod->forward(m_statePrev);
		
		m_state.copy(m_inpMod->output()).addMM(m_memMod->output());
		return m_outMod->forward(m_state);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		if(m_resetGrad)
		{
			m_resetGrad = false;
			m_stateGrad.fill(0);
		}
		
		m_outMod->backward(m_state, outGrad);
		m_outMod->inGrad().addMM(m_stateGrad);
		m_stateGrad.copy(m_memMod->backward(m_statePrev, m_outMod->inGrad()));
		return m_inpMod->backward(input, m_outMod->inGrad());
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_outMod->output();
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inpMod->inGrad();
	}
	
	/// Set the input shape of this module, including batch.
	virtual Recurrent &inputs(const Storage<size_t> &dims) override
	{
		m_inpMod->inputs(dims);
		return batch(dims[0]);
	}
	
	/// Set the output shape of this module, including batch.
	virtual Recurrent &outputs(const Storage<size_t> &dims) override
	{
		m_memMod->outputs(dims);
		m_outMod->outputs(dims);
		m_state.resize(dims);
		m_statePrev.resize(dims);
		m_stateGrad.resize(dims);
		return batch(dims[0]);
	}
	
	/// Set the batch size of this module.
	virtual Recurrent &batch(size_t bats) override
	{
		Container<T>::batch(bats);
		m_state.resizeDim(0, bats);
		m_statePrev.resizeDim(0, bats);
		m_stateGrad.resizeDim(0, bats);
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor<T> *> innerState() override
	{
		Storage<Tensor<T> *> states = Container<T>::innerState();
		states.push_back(&m_state);
		states.push_back(&m_statePrev);
		return states;
	}
private:
	Module<T> *m_inpMod;
	Module<T> *m_memMod;
	Module<T> *m_outMod;
	
	Tensor<T> m_state;
	Tensor<T> m_statePrev;
	Tensor<T> m_stateGrad;
	
	bool m_resetGrad;
};

}

#endif
