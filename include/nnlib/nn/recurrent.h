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
		m_outMod(new Sequential<T>(new Linear<T>(outs, outs, bats), new TanH<T>())),
		m_state(bats, outs),
		m_statePrev(bats, outs),
		m_stateGrad(bats, outs),
		m_resetGrad(true)
	{
		Container<T>::add(m_inpMod);
		Container<T>::add(m_memMod);
		Container<T>::add(m_outMod);
		forget();
	}
	
	Recurrent(size_t outs = 0) :
		m_inpMod(new Linear<T>(0, outs, 1)),
		m_memMod(new Linear<T>(outs, outs, 1)),
		m_outMod(new Sequential<T>(new Linear<T>(outs, outs, 1), new TanH<T>())),
		m_state(1, outs),
		m_statePrev(1, outs),
		m_stateGrad(1, outs),
		m_resetGrad(true)
	{
		Container<T>::add(m_inpMod);
		Container<T>::add(m_memMod);
		Container<T>::add(m_outMod);
		forget();
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
		NNAssertEquals(m_inpMod->outputs().size(), 2, "Expected matrix input!");
		NNAssertEquals(m_memMod->outputs().size(), 2, "Expected matrix input!");
		NNAssertEquals(m_outMod->outputs().size(), 2, "Expected matrix input!");
		Container<T>::add(m_inpMod);
		Container<T>::add(m_memMod);
		Container<T>::add(m_outMod);
		forget();
	}
	
	Recurrent(const Recurrent &module) :
		m_inpMod(copy(module.m_inpMod)),
		m_memMod(copy(module.m_memMod)),
		m_outMod(copy(module.m_outMod)),
		m_state(module.m_state.copy()),
		m_statePrev(module.m_statePrev.copy()),
		m_stateGrad(module.m_stateGrad.copy()),
		m_resetGrad(module.m_resetGrad)
	{
		Container<T>::add(m_inpMod);
		Container<T>::add(m_memMod);
		Container<T>::add(m_outMod);
	}
	
	Recurrent &operator=(const Recurrent &module)
	{
		m_inpMod	= copy(module.m_inpMod);
		m_memMod	= copy(module.m_memMod);
		m_outMod	= copy(module.m_outMod);
		m_state		= module.m_state.copy();
		m_statePrev	= module.m_statePrev.copy();
		m_stateGrad	= module.m_stateGrad.copy();
		m_resetGrad	= module.m_resetGrad;
		return *this;
	}
	
	// MARK: Container methods
	
	/// Cannot add a component to this container.
	virtual Recurrent &add(Module<T> *component) override
	{
		throw Error("Cannot add components to a recurrent module!");
	}
	
	/// Cannot remove a component from this container.
	virtual Module<T> *remove(size_t) override
	{
		throw Error("Cannot remove components from a recurrent module!");
	}
	
	/// Cannot remove a component from this container.
	virtual Recurrent &clear() override
	{
		throw Error("Cannot remove components from a recurrent module!");
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_resetGrad = true;
		
		m_statePrev.copy(m_state);
		m_inpMod->forward(input);
		m_memMod->forward(m_statePrev);
		
		m_state.copy(m_inpMod->output()).addM(m_memMod->output());
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
		m_outMod->inGrad().addM(m_stateGrad);
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
		m_resetGrad = true;
		return *this;
	}
	
	/// A vector of tensors filled with (views of) this module's internal state.
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states = Container<T>::stateList();
		states.push_back(&m_state);
		states.push_back(&m_statePrev);
		return states;
	}
	
	/// Reset the internal state of this module.
	virtual Recurrent &forget() override
	{
		m_state.fill(0);
		m_resetGrad = true;
		return *this;
	}
	
	
	/// \brief Write to an archive.
	///
	/// \param ar The archive to which to write.
	template <typename Archive>
	void save(Archive &ar) const
	{
		ar(m_inpMod, m_memMod, m_outMod);
	}
	
	/// \brief Read from an archive.
	///
	/// \param ar The archive from which to read.
	template <typename Archive>
	void load(Archive &ar)
	{
		Container<T>::clear();
		ar(m_inpMod, m_memMod, m_outMod);
		Container<T>::add(m_inpMod);
		Container<T>::add(m_memMod);
		Container<T>::add(m_outMod);
		
		m_state.resize(m_outMod->outputs());
		m_statePrev.resize(m_outMod->outputs());
		m_stateGrad.resize(m_outMod->outputs());
		m_resetGrad = true;
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

NNRegisterType(Recurrent<float>, Module<float>);
NNRegisterType(Recurrent<double>, Module<double>);

#endif
