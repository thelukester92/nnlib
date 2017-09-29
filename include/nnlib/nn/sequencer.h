#ifndef NN_SEQUENCER_H
#define NN_SEQUENCER_H

#include "module.h"

namespace nnlib
{

/// \brief Allows an extra "sequence" dimension when passing in inputs to a module.
///
/// Inputs will be passed through to the inner module one at a time and backpropagated
/// in reverse order, essentially abstracting away BPTT.
template <typename T = double>
class Sequencer : public Module<T>
{
public:
	Sequencer(Module<T> *module) :
		m_module(module)
	{}
	
	Sequencer(const Sequencer &module) :
		m_module(module.m_module->copy())
	{}
	
	Sequencer(const Serialized &node) :
		m_module(node.get<Module<T> *>("module"))
	{}
	
	Sequencer &operator=(Sequencer module)
	{
		swap(*this, module);
		return *this;
	}
	
	friend void swap(Sequencer &a, Sequencer &b)
	{
		using std::swap;
		swap(a.m_module, b.m_module);
	}
	
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
		delete m_module;
		m_module = module;
		return *this;
	}
	
	virtual void training(bool training = true) override
	{
		m_module->training(training);
	}
	
	virtual void forget() override
	{
		m_module->forget();
	}
	
	/// MARK: Serialization
	
	virtual void save(Serialized &node) const override
	{
		node.set("module", m_module);
	}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_module->forward(input.select(0, 0));
		
		m_output.resize(Storage<size_t>({ input.size(0) }).append(m_module->output().shape()));
		m_output.select(0, 0).copy(m_module->output());
		
		m_states.resize(Storage<size_t>({ input.size(0) }).append(m_module->state().shape()));
		m_states.select(0, 0).copy(m_module->state());
		
		for(size_t i = 1, end = input.size(0); i < end; ++i)
		{
			m_output.select(0, i).copy(m_module->forward(input.select(0, i)));
			m_states.select(0, i).copy(m_module->state());
		}
		
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.size(0), outGrad.size(0), "Incompatible input and outGrad!");
		NNAssertEquals(input.size(0), m_output.size(0), "Sequencer::forward must be called first!");
		m_inGrad.resize(input.shape());
		
		for(int i = input.size(0) - 1; i >= 0; --i)
		{
			m_module->state().copy(m_states.select(0, i));
			m_inGrad.select(0, i).copy(m_module->backward(input.select(0, i), outGrad.select(0, i)));
		}
		
		return m_inGrad;
	}
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> paramsList() override
	{
		return m_module->paramsList();
	}
	
	virtual Storage<Tensor<T> *> gradList() override
	{
		return m_module->gradList();
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
private:
	Module<T> *m_module;
	Tensor<T> m_states;
};

}

NNRegisterType(Sequencer, Module);

#endif
