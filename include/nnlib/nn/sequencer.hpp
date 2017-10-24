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
	Sequencer(Module<T> *module, bool reverse = false) :
		m_module(module),
		m_reverse(reverse)
	{}
	
	Sequencer(const Sequencer &module) :
		m_module(module.m_module->copy()),
		m_reverse(module.m_reverse)
	{}
	
	Sequencer(const Serialized &node) :
		m_module(node.get<Module<T> *>("module")),
		m_reverse(node.get<bool>("reverse"))
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
		swap(a.m_reverse, b.m_reverse);
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
	
	/// Check whether this sequencer reverses the input sequence.
	bool isReversed()
	{
		return m_reverse;
	}
	
	/// Set whether this sequencer reverses the input sequence.
	Sequencer &reverse(bool reverse = true)
	{
		m_reverse = reverse;
		return *this;
	}
	
	/// Begin a sequence. This is automatically called by forward.
	void startForward(const Tensor<T> &first, size_t sequenceLength)
	{
		m_module->forward(first);
		
		m_output.resize(Storage<size_t>({ sequenceLength }).append(m_module->output().shape()));
		if(m_reverse)
			m_output.select(0, sequenceLength - 1).copy(m_module->output());
		else
			m_output.select(0, 0).copy(m_module->output());
		
		m_states.resize(Storage<size_t>({ sequenceLength }).append(m_module->state().shape()));
		if(m_reverse)
			m_states.select(0, sequenceLength - 1).copy(m_module->state());
		else
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
		Module<T>::forget();
		m_module->forget();
	}
	
	/// MARK: Serialization
	
	virtual void save(Serialized &node) const override
	{
		node.set("module", m_module);
		node.set("reverse", m_reverse);
	}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		if(m_reverse)
		{
			size_t len = input.size(0);
			startForward(input.select(0, len - 1), len);
			for(size_t i = len - 1; i > 0; --i)
				stepForward(input.select(0, i - 1), i - 1);
		}
		else
		{
			startForward(input.select(0, 0), input.size(0));
			for(size_t i = 1, end = input.size(0); i < end; ++i)
				stepForward(input.select(0, i), i);
		}
		
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.size(0), outGrad.size(0), "Incompatible input and outGrad!");
		NNAssertEquals(input.size(0), m_output.size(0), "Sequencer::forward must be called first!");
		m_inGrad.resize(input.shape());
		
		if(m_reverse)
		{
			for(size_t i = 0, len = input.size(0); i < len; ++i)
				stepBackward(input.select(0, i), outGrad.select(0, i), i);
		}
		else
		{
			for(size_t i = input.size(0); i > 0; --i)
				stepBackward(input.select(0, i - 1), outGrad.select(0, i - 1), i - 1);
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
	
	virtual Storage<Tensor<T> *> stateList() override
	{
		return Module<T>::stateList().append(m_module->stateList()).push_back(&m_states);
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
private:
	Module<T> *m_module;
	Tensor<T> m_states;
	bool m_reverse;
};

}

NNRegisterType(Sequencer, Module);

#endif
