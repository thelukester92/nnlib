#ifndef DROPCONNECT_H
#define DROPCONNECT_H

#include "module.h"

namespace nnlib
{

/// A module decorator that randomly drops parameters with a given probability.
template <typename T = double>
class DropConnect : public Module<T>
{
public:
	DropConnect(Module<T> *module, T dropProbability = 0.1) :
		m_module(module),
		m_dropProbability(dropProbability),
		m_training(true)
	{
		NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
		NNAssertLessThan(dropProbability, 1, "Expected a probability!");
	}
	
	DropConnect(const DropConnect &module) :
		m_module(module.m_module->copy()),
		m_dropProbability(module.m_dropProbability),
		m_training(module.m_training)
	{}
	
	DropConnect(const Serialized &node) :
		m_module(node.get<Module<T> *>("module")),
		m_dropProbability(node.get<T>("dropProbability")),
		m_training(node.get<bool>("training"))
	{}
	
	DropConnect &operator=(DropConnect module)
	{
		swap(*this, module);
		return *this;
	}
	
	virtual ~DropConnect()
	{
		delete m_module;
	}
	
	friend void swap(DropConnect &a, DropConnect &b)
	{
		using std::swap;
		swap(a.m_module, b.m_module);
		swap(a.m_dropProbability, b.m_dropProbability);
		swap(a.m_training, b.m_training);
	}
	
	/// Get the module this is decorating.
	Module<T> &module()
	{
		return *m_module;
	}
	
	/// Set the module this is decorating.
	DropConnect &module(Module<T> *module)
	{
		delete m_module;
		m_module = module;
	}
	
	/// Get the probability that an output is not dropped.
	T dropProbability() const
	{
		return m_dropProbability;
	}
	
	/// Set the probability that an output is not dropped.
	DropConnect &dropProbability(T dropProbability)
	{
		NNAssertGreaterThanOrEquals(dropProbability, 0, "Expected a probability!");
		NNAssertLessThan(dropProbability, 1, "Expected a probability!");
		m_dropProbability = dropProbability;
		return *this;
	}
	
	virtual void training(bool training = true) override
	{
		m_training = training;
		m_module->training(training);
	}
	
	virtual void forget() override
	{
		m_module->forget();
	}
	
	// MARK: Serialization
	
	virtual void save(Serialized &node) const override
	{
		node.set("module", m_module);
		node.set("dropProbability", m_dropProbability);
		node.set("training", m_training);
	}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		if(m_training)
		{
			m_output.resize(input.shape());
			m_mask.resize(m_module->params().shape());
			m_backup.resize(m_module->params().shape());
			m_backup.copy(m_module->params());
			m_module->params().pointwiseProduct(m_mask.bernoulli(1 - m_dropProbability));
			m_output = m_module->forward(input);
		}
		else
			m_output = m_module->forward(input).scale(1 - m_dropProbability);
		
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		if(m_training)
		{
			m_inGrad = m_module->backward(input, outGrad);
			m_module->params().copy(m_backup);
		}
		else
			m_inGrad = m_module->backward(input, outGrad).scale(1 - m_dropProbability);
		
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
		return Module<T>::stateList().append({ &m_mask, &m_backup });
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
private:
	Module<T> *m_module;
	Tensor<T> m_backup;
	Tensor<T> m_mask;
	T m_dropProbability;
	bool m_training;
};

}

NNRegisterType(DropConnect, Module);

#endif
