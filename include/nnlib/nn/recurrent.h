#ifndef NN_RECURRENT_H
#define NN_RECURRENT_H

#include "container.h"
#include "sequential.h"
#include "linear.h"
#include "tanh.h"

namespace nnlib
{

/// A simple recurrent module.
class Recurrent : public Container
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
		return "recurrent";
	}
	
	Recurrent(size_t inps, size_t outs, size_t bats = 1) :
		m_inpMod(new Linear(inps, outs, bats)),
		m_memMod(new Linear(outs, outs, bats)),
		m_outMod(new Sequential(new Linear(outs, outs, bats), new TanH())),
		m_state(bats, outs),
		m_statePrev(bats, outs),
		m_stateGrad(bats, outs),
		m_resetGrad(true)
	{
		Container::add(m_inpMod);
		Container::add(m_memMod);
		Container::add(m_outMod);
		forget();
	}
	
	Recurrent(size_t outs) :
		m_inpMod(new Linear(0, outs, 1)),
		m_memMod(new Linear(outs, outs, 1)),
		m_outMod(new Sequential(new Linear(outs, outs, 1), new TanH())),
		m_state(1, outs),
		m_statePrev(1, outs),
		m_stateGrad(1, outs),
		m_resetGrad(true)
	{
		Container::add(m_inpMod);
		Container::add(m_memMod);
		Container::add(m_outMod);
		forget();
	}
	
	Recurrent(Module *inpMod, Module *memMod, Module *outMod) :
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
		Container::add(m_inpMod);
		Container::add(m_memMod);
		Container::add(m_outMod);
		forget();
	}
	
	Recurrent &forget()
	{
		m_state.fill(0);
		m_resetGrad = true;
		return *this;
	}
	
	// MARK: Container methods
	
	/// Cannot add a component to this container.
	virtual Recurrent &add(Module *component) override
	{
		throw std::runtime_error("Cannot add components to a recurrent module!");
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor &forward(const Tensor &input) override
	{
		m_resetGrad = true;
		
		m_statePrev.copy(m_state);
		m_inpMod->forward(input);
		m_memMod->forward(m_statePrev);
		
		m_state.copy(m_inpMod->output()).addMM(m_memMod->output());
		return m_outMod->forward(m_state);
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor &backward(const Tensor &input, const Tensor &outGrad) override
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
	virtual Tensor &output() override
	{
		return m_outMod->output();
	}
	
	/// Cached input gradient.
	virtual Tensor &inGrad() override
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
		Container::batch(bats);
		m_state.resizeDim(0, bats);
		m_statePrev.resizeDim(0, bats);
		m_stateGrad.resizeDim(0, bats);
		m_resetGrad = true;
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor *> stateList() override
	{
		Storage<Tensor *> states = Container::stateList();
		states.push_back(&m_state);
		states.push_back(&m_statePrev);
		return states;
	}
private:
	Module *m_inpMod;
	Module *m_memMod;
	Module *m_outMod;
	
	Tensor m_state;
	Tensor m_statePrev;
	Tensor m_stateGrad;
	
	bool m_resetGrad;
};

}

#endif
