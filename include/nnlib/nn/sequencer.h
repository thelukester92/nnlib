#ifndef NN_SEQUENCER_H
#define NN_SEQUENCER_H

#include "container.h"

namespace nnlib
{

/// A container module that accepts sequential inputs and processes them
/// one at a time, essentially abstracting away BPTT.
/// Input should be a 3D tensor: sequence length X batch size X inputs.
/// Output should be a 3D tensor: sequence length X batch size X outputs.
template <typename T = double>
class Sequencer : public Container<T>
{
public:
	using Container<T>::inputs;
	using Container<T>::outputs;
	using Container<T>::batch;
	using Container<T>::add;
	
	Sequencer(Module<T> *module, size_t seqLen = 0) :
		m_module(module),
		m_state(Tensor<T>::flatten(module->innerState())),
		m_states(seqLen, m_state.size(0))
	{
		Storage<size_t> inps = { seqLen };
		for(size_t size : m_module->inputs())
			inps.push_back(size);
		m_inGrad.resize(inps);
		
		Storage<size_t> outs = { seqLen };
		for(size_t size : m_module->outputs())
			outs.push_back(size);
		m_output.resize(outs);
		
		add(module);
	}
	
	Sequencer &seqLen(size_t seqLen)
	{
		m_inGrad.resizeDim(0, seqLen);
		m_output.resizeDim(0, seqLen);
		m_states.resizeDim(0, seqLen);
		return *this;
	}
	
	size_t seqLen() const
	{
		return m_states.size(0);
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssert(input.shape() == m_inGrad.shape(), "Incompatible input! Must be seqLen X batch X inputs!");
		
		/// \todo only reset the state if required
		m_state.fill(0);
		
		for(size_t i = 0, end = input.size(0); i < end; ++i)
		{
			m_output.select(0, i).copy(m_module->forward(input.select(0, i)));
			m_states.select(0, i).copy(m_state);
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssert(input.shape() == m_inGrad.shape(), "Incompatible input! Must be seqLen X batch X inputs!");
		NNAssert(outGrad.shape() == m_output.shape(), "Incompatible outGrad! Must be seqLen X batch X outputs!");
		
		for(size_t i = input.size(0) - 1; i > 0; --i)
		{
			m_state.copy(m_states.select(0, i - 1));
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
	
	/// Set the input shape of this module, including batch.
	virtual Sequencer &inputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 3, "Sequencer expects a seqLen x batch x inputs input tensor!");
		
		Storage<size_t> newDims = dims;
		newDims.erase(0);
		m_module->inputs(newDims);
		
		m_inGrad.resize(dims);
		m_output.resizeDim(0, dims[0]);
		m_output.resizeDim(1, dims[1]);
		
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	virtual Sequencer &outputs(const Storage<size_t> &dims) override
	{
		NNAssert(dims.size() == 3, "Sequencer expects a seqLen x batch x outputs output tensor!");
		
		Storage<size_t> newDims = dims;
		newDims.erase(0);
		m_module->outputs(newDims);
		
		m_output.resize(dims);
		m_inGrad.resizeDim(0, dims[0]);
		m_inGrad.resizeDim(1, dims[1]);
		
		return *this;
	}
	
	/// Set the batch size of this module.
	virtual Sequencer &batch(size_t bats) override
	{
		
		return *this;
	}
	
private:
	Module<T> *m_module;
	
	Tensor<T> m_output;
	Tensor<T> m_inGrad;
	
	Tensor<T> m_state;
	Tensor<T> m_states;
};

}

#endif
