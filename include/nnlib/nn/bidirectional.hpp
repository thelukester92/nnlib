#ifndef NN_BIDIRECTIONAL_HPP
#define NN_BIDIRECTIONAL_HPP

#include "sequencer.hpp"

namespace nnlib
{

/// \brief Allows an extra "sequence" dimension when passing in inputs to a module; bidirectional.
///
/// Combines two Sequencers (one forward, one backward) using concatenation for merging the output.
template <typename T = double>
class Bidirectional : public Module<T>
{
public:
	Bidirectional(Module<T> *module, size_t concatDim = (size_t) -1) :
		m_fModule(new Sequencer<T>(module)),
		m_bModule(new Sequencer<T>(module->copy())),
		m_concatDim(concatDim)
	{}
	
	Bidirectional(const Bidirectional &module) :
		m_fModule(module.m_fModule->copy()),
		m_bModule(module.m_bModule->copy()),
		m_concatDim(module.m_concatDim)
	{}
	
	Bidirectional(const Serialized &node) :
		m_fModule(node.get<Module<T> *>("fModule")),
		m_bModule(node.get<Module<T> *>("bModule")),
		m_concatDim(node.get<Module<T> *>("concatDim"))
	{}
	
	~Bidirectional()
	{
		delete m_fModule;
		delete m_bModule;
	}
	
	Bidirectional &operator=(Bidirectional module)
	{
		swap(*this, module);
		return *this;
	}
	
	friend void swap(Bidirectional &a, Bidirectional &b)
	{
		using std::swap;
		swap(a.m_fModule, b.m_fModule);
		swap(a.m_bModule, b.m_bModule);
		swap(a.m_concatDim, b.m_concatDim);
	}
	
	/// Get the (forward) module used by this sequencer.
	Module<T> &module()
	{
		return m_fModule->module();
	}
	
	/// Set the forward module used by this sequencer.
	Bidirectional &module(Module<T> *module)
	{
		m_fModule->module(module);
		m_bModule->module(module->copy());
		return *this;
	}
	
	size_t concatDim() const
	{
		return m_concatDim;
	}
	
	Bidirectional &concatDim(size_t dim)
	{
		m_concatDim = dim;
		return *this;
	}
	
	virtual void training(bool training = true) override
	{
		m_fModule->training(training);
		m_bModule->training(training);
	}
	
	virtual void forget() override
	{
		m_fModule->forget();
		m_bModule->forget();
	}
	
	/// MARK: Serialization
	
	virtual void save(Serialized &node) const override
	{
		node.set("fModule", m_fModule);
		node.set("bModule", m_bModule);
		node.set("concatDim", m_concatDim);
	}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		size_t len = input.size(0);
		
		m_fModule->startForward(input.select(0, 0), len);
		m_bModule->startForward(input.select(0, len - 1), len);
		
		for(size_t i = 1; i < len; ++i)
		{
			m_fModule->stepForward(input.select(0, i), i);
			m_bModule->stepForward(input.select(0, len - 1 - i), len - 1 - i);
		}
		
		Storage<Tensor<T> *> outputs = { &m_fModule->output(), &m_bModule->output() };
		if(!m_output.sharedWith(outputs))
		{
			m_concatDim = std::min(m_concatDim, m_fModule->output().dims() - 1);
			m_output = Tensor<T>::concatenate(outputs, m_concatDim);
		}
		
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.size(0), outGrad.size(0), "Incompatible input and outGrad!");
		NNAssertEquals(input.size(0), m_output.size(0), "Sequencer::forward must be called first!");
		m_inGrad.resize(input.shape());
		
		size_t len = input.size(0);
		for(int i = len - 1; i >= 0; --i)
		{
			m_fModule->stepBackward(input.select(0, i), outGrad.select(0, i), i);
			m_bModule->stepBackward(input.select(0, len - 1 - i), outGrad.select(0, i), i);
		}
		
		m_inGrad.resize(m_fModule->inGrad().shape());
		m_inGrad.add(m_fModule->inGrad());
		m_inGrad.add(m_bModule->inGrad());
		
		return m_inGrad;
	}
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> paramsList() override
	{
		return m_fModule->paramsList().append(m_bModule->paramsList());
	}
	
	virtual Storage<Tensor<T> *> gradList() override
	{
		return m_fModule->gradList().append(m_bModule->gradList());
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
private:
	Module<T> *m_fModule, m_bModule;
};

}

NNRegisterType(Bidirectional, Module);

#endif
