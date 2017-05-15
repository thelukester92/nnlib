#ifndef NN_SEQUENCER_H
#define NN_SEQUENCER_H

#include "container.h"

namespace nnlib
{

/// A container module that accepts sequential inputs and processes them
/// one at a time, essentially abstracting away BPTT.
/// Input should be a 3D tensor: sequence X batch size X inputs.
/// Output should be a 3D tensor: sequence X batch size X outputs.
class Sequencer : public Container
{
public:
	using Container::inputs;
	using Container::outputs;
	using Container::batch;
	using Container::add;
	
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "sequencer";
	}
	
	Sequencer(Module *module, size_t sequenceLength = 1) :
		m_module(module),
		m_state(module->state()),
		m_states(sequenceLength, m_state.size(0))
	{
		Storage<size_t> inps = { sequenceLength };
		for(size_t size : m_module->inputs())
			inps.push_back(size);
		m_inGrad.resize(inps);
		
		Storage<size_t> outs = { sequenceLength };
		for(size_t size : m_module->outputs())
			outs.push_back(size);
		m_output.resize(outs);
		
		add(module);
	}
	
	/// Get the module used by this sequencer.
	Module &module()
	{
		return *m_module;
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
	
	Sequencer &forget()
	{
		m_state.fill(0);
		return *this;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor &forward(const Tensor &input) override
	{
		NNAssert(input.shape() == m_inGrad.shape(), "Incompatible input! Must be sequence x batch x inputs!");
		
		for(size_t i = 0, end = input.size(0); i < end; ++i)
		{
			m_output.select(0, i).copy(m_module->forward(input.select(0, i)));
			m_states.select(0, i).copy(m_state);
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor &backward(const Tensor &input, const Tensor &outGrad) override
	{
		NNAssert(input.shape() == m_inGrad.shape(), "Incompatible input! Must be sequence x batch x inputs!");
		NNAssert(outGrad.shape() == m_output.shape(), "Incompatible outGrad! Must be sequence x batch x outputs!");
		
		for(int i = input.size(0) - 1; i >= 0; --i)
		{
			m_state.copy(m_states.select(0, i));
			m_inGrad.select(0, i).copy(m_module->backward(input.select(0, i), outGrad.select(0, i)));
		}
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including batch.
	virtual Sequencer &inputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 3, "Sequencer expects a sequence x batch x inputs input tensor!");
		
		Storage<size_t> newDims = dims;
		newDims.erase(0);
		m_module->inputs(newDims);
		
		m_inGrad.resize(dims);
		m_output.resizeDim(0, dims[0]);
		m_output.resizeDim(1, dims[1]);
		
		m_module->state();
		m_states.resizeDim(1, m_state.size(0));
		
		return *this;
	}
	
	/// Safely (never reset weights) set the input shape of this module.
	virtual Sequencer &safeInputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 3, "Sequencer expects a sequence x batch x inputs input tensor!");
		
		if(dims[2] == 0)
		{
			inputs(dims);
		}
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
		NNAssert(dims.size() == 3, "Sequencer expects a sequenceLength x batch x outputs output tensor!");
		
		Storage<size_t> newDims = dims;
		newDims.erase(0);
		m_module->outputs(newDims);
		
		m_output.resize(dims);
		m_inGrad.resizeDim(0, dims[0]);
		m_inGrad.resizeDim(1, dims[1]);
		
		m_module->state();
		m_states.resizeDim(1, m_state.size(0));
		
		return *this;
	}
	
	/// Safely (never reset weights) set the output shape of this module.
	virtual Sequencer &safeOutputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 3, "Sequencer expects a sequence x batch x inputs input tensor!");
		
		if(dims[2] == 0)
		{
			outputs(dims);
		}
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
		m_states.resizeDim(1, m_state.size(0));
		
		return *this;
	}
	
	/// Get the batch size of this module.
	virtual size_t batch() const override
	{
		return m_output.size(1);
	}
	
private:
	Module *m_module;
	
	Tensor m_output;
	Tensor m_inGrad;
	
	Tensor &m_state;
	Tensor m_states;
};

}

#endif
