#ifndef NN_SEQUENCER_HPP
#define NN_SEQUENCER_HPP

#include "module.hpp"

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
	
	~Sequencer()
	{
		delete m_module;
	}
	
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
	
	/// Begin a sequence. This is automatically called by forward.
	void startForward(const Tensor<T> &first, size_t sequenceLength)
	{
		m_module->forward(first);
		
		m_output.resize(Storage<size_t>({ sequenceLength }).append(m_module->output().shape()));
		m_output.select(0, 0).copy(m_module->output());
		
		m_states.resize(Storage<size_t>({ sequenceLength }).append(m_module->state().shape()));
		m_states.select(0, 0).copy(m_module->state());
	}
	
	/// Forward the next sample in the sequence. This is automatically called by forward.
	void stepForward(const Tensor<T> &singleInput, size_t i)
	{
		m_output.select(0, i).copy(m_module->forward(singleInput));
		m_states.select(0, i).copy(m_module->state());
	}
	
	/// Backward the next sample in the sequence. This is automatically called by backward.
	void stepBackward(const Tensor<T> &singleInput, const Tensor<T> &singleOutGrad, size_t i)
	{
		m_module->state().copy(m_states.select(0, i));
		m_inGrad.select(0, i).copy(m_module->backward(singleInput, singleOutGrad));
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
		startForward(input.select(0, 0), input.size(0));
		for(size_t i = 1, end = input.size(0); i < end; ++i)
			stepForward(input.select(0, i), i);
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.size(0), outGrad.size(0), "Incompatible input and outGrad!");
		NNAssertEquals(input.size(0), m_output.size(0), "Sequencer::forward must be called first!");
		m_inGrad.resize(input.shape());
		
		for(int i = input.size(0) - 1; i >= 0; --i)
			stepBackward(input.select(0, i), outGrad.select(0, i), i);
		
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
