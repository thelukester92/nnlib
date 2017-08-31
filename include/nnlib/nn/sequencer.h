#ifndef NN_SEQUENCER_H
#define NN_SEQUENCER_H

#include "container.h"

namespace nnlib
{

/// A container module that accepts sequential inputs and processes them
/// one at a time, essentially abstracting away BPTT.
/// Input should be a 3D tensor: sequence X batch size X inputs.
/// Output should be a 3D tensor: sequence X batch size X outputs.
template <typename T = double>
class Sequencer : public Container<T>
{
public:
	using Container<T>::inputs;
	using Container<T>::outputs;
	using Container<T>::batch;
	using Container<T>::add;
	
	Sequencer(Module<T> *module = nullptr, size_t sequenceLength = 1) :
		m_module(module),
		m_state(module == nullptr ? nullptr : &module->state()),
		m_states(sequenceLength, module == nullptr ? 0 : m_state->size(0))
	{
		if(module != nullptr)
		{
			Storage<size_t> inps = { sequenceLength };
			for(size_t size : m_module->inputs())
				inps.push_back(size);
			m_inGrad.resize(inps);
			
			Storage<size_t> outs = { sequenceLength };
			for(size_t size : m_module->outputs())
				outs.push_back(size);
			m_output.resize(outs);
		}
		Container<T>::add(module);
	}
	
	Sequencer(const Sequencer &module) :
		m_module(module.m_module == nullptr ? nullptr : module.m_module->copy()),
		m_output(module.m_output.copy()),
		m_inGrad(module.m_inGrad.copy()),
		m_state(m_module == nullptr ? nullptr : &m_module->state()),
		m_states(module.m_states.copy())
	{
		Container<T>::add(m_module);
	}
	
	Sequencer &operator=(const Sequencer &module)
	{
		m_module	= module.m_module == nullptr ? nullptr : module.m_module->copy();
		m_state		= m_module == nullptr ? nullptr : &m_module->state();
		m_states	= module.m_states.copy();
		m_inGrad	= module.m_inGrad.copy();
		m_output	= module.m_output.copy();
		
		Container<T>::clear();
		Container<T>::add(m_module);
		
		return *this;
	}
	
	// MARK: Container methods
	
	/// Cannot add a component to this container.
	virtual Sequencer &add(Module<T> *) override
	{
		throw Error("Cannot add components to a Sequencer module!");
	}
	
	/// Cannot remove a component from this container.
	virtual Module<T> *remove(size_t) override
	{
		throw Error("Cannot remove components from a Sequencer module!");
	}
	
	/// Cannot remove a component from this container.
	virtual Sequencer &clear() override
	{
		throw Error("Cannot remove components from a Sequencer module!");
	}
	
	// MARK: Sequencer methods
	
	/// Get the module used by this sequencer.
	Module<T> &module()
	{
		return *m_module;
	}
	
	/// \brief Set the module used by this sequencer.
	///
	/// This also deletes the module previously used by this sequencer.
	Sequencer &module(Module<T> *module)
	{
		m_module = module;
		Container<T>::clear();
		Container<T>::add(m_module);
		return *this;
	}
	
	/// Set the length of the sequence this module uses.
	Sequencer &sequenceLength(size_t sequenceLength)
	{
		m_inGrad.resizeDim(0, sequenceLength);
		m_output.resizeDim(0, sequenceLength);
		m_states.resizeDim(0, sequenceLength);
		return *this;
	}
	
	/// Get the length of the sequence this module uses.
	size_t sequenceLength() const
	{
		return m_states.size(0);
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
		
		for(size_t i = 0, end = input.size(0); i < end; ++i)
		{
			m_output.select(0, i).copy(m_module->forward(input.select(0, i)));
			m_states.select(0, i).copy(*m_state);
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.shape(), m_inGrad.shape(), "Incompatible input!");
		NNAssertEquals(outGrad.shape(), m_output.shape(), "Incompatible output!");
		
		for(int i = input.size(0) - 1; i >= 0; --i)
		{
			m_state->copy(m_states.select(0, i));
			m_inGrad.select(0, i).copy(m_module->backward(input.select(0, i), outGrad.select(0, i)));
		}
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including sequence length and batch.
	virtual Sequencer &inputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 3, "Expected 3D input!");
		
		Storage<size_t> newDims = dims;
		newDims.erase(0);
		m_module->inputs(newDims);
		
		m_inGrad.resize(dims);
		return sequenceLength(dims[0]).batch(dims[1]);
	}
	
	/// Safely (never reset weights) set the input shape of this module.
	virtual Sequencer &safeInputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 3, "Expected 3D input!");
		
		if(inGrad().size(2) == 0)
			inputs(dims);
		else
		{
			sequenceLength(dims[0]);
			batch(dims[1]);
		}
		
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	virtual Sequencer &outputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 3, "Expected 3D output!");
		
		Storage<size_t> newDims = dims;
		newDims.erase(0);
		m_module->outputs(newDims);
		
		m_output.resize(dims);
		m_inGrad.resizeDim(0, dims[0]);
		m_inGrad.resizeDim(1, dims[1]);
		
		m_module->state();
		m_states.resizeDim(1, m_state->size(0));
		
		return *this;
	}
	
	/// Safely (never reset weights) set the output shape of this module.
	virtual Sequencer &safeOutputs(const Storage<size_t> &dims) override
	{
		NNAssertEquals(dims.size(), 3, "Expected 3D output!");
		
		if(output().size(2) == 0)
			outputs(dims);
		else
		{
			sequenceLength(dims[0]);
			batch(dims[1]);
		}
		
		return *this;
	}
	
	/// Set the batch size of this module.
	virtual Sequencer &batch(size_t bats) override
	{
		m_module->batch(bats);
		m_output.resizeDim(1, bats);
		m_inGrad.resizeDim(1, bats);
		
		/// \note this is stupid, the way I re-flatten state
		m_module->state();
		m_states.resizeDim(1, m_state->size(0));
		
		return *this;
	}
	
	/// Get the batch size of this module.
	virtual size_t batch() const override
	{
		return m_output.size(1);
	}
	
	/// Save to a serialized node.
	virtual void save(Serialized &node) const override
	{
		node.set("module", m_module);
		node.set("sequenceLength", sequenceLength());
	}
	
	/// Load from a serialized node.
	virtual void load(const Serialized &node) override
	{
		module(node.get<Module<T> *>("module"));
		size_t seqLen = node.get<size_t>("sequenceLength");
		
		m_state = &m_module->state();
		m_states.resize(seqLen, m_state->size(0));
		
		Storage<size_t> inps = { seqLen };
		for(size_t size : m_module->inputs())
			inps.push_back(size);
		m_inGrad.resize(inps);
		
		Storage<size_t> outs = { seqLen };
		for(size_t size : m_module->outputs())
			outs.push_back(size);
		m_output.resize(outs);
	}
	
private:
	Module<T> *m_module;
	
	Tensor<T> m_output;
	Tensor<T> m_inGrad;
	
	Tensor<T> *m_state;
	Tensor<T> m_states;
};

}

NNRegisterType(Sequencer<float>, Module<float>);
NNRegisterType(Sequencer<double>, Module<double>);

#endif
