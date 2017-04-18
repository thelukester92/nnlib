#ifndef NN_SEQUENTIAL_H
#define NN_SEQUENTIAL_H

#include "container.h"

namespace nnlib
{

/// A standard feed-forward neural network module.
template <typename T = double>
class Sequential : public Container<T>
{
using Container<T>::components;
using Container<T>::m_components;
public:
	Sequential() {}
	
	template <typename ... Ms>
	Sequential(Module<T> *component, Ms *...components)
	{
		add(component, components...);
	}
	
	template <typename ... Ms>
	void add(Module<T> *component, Ms *...more)
	{
		add(component);
		add(more...);
	}
	
	// MARK: Container methods
	
	/// Add a component to this container, enforcing compatibility.
	virtual void add(Module<T> *component) override
	{
		m_components.push_back(component);
		if(components() > 1)
		{
			component->resizeInput(m_components[m_components.size() - 2]->output().shape());
		}
	}
	
	/// Remove and return a specific component from this container, enforcing compatibility.
	virtual Module<T> *remove(size_t index) override
	{
		Module<T> *comp = m_components[index];
		m_components.erase(index);
		
		if(index > 0)
		{
			for(size_t i = index, j = components(); i < j; ++i)
			{
				m_components[i]->resizeInput(m_components[i - 1]->output().shape());
			}
		}
		
		return comp;
	}
	
	// MARK: Module methods
	
	/// Change the input dimensions of this module, enforcing compatibility.
	virtual void resizeInput(const Storage<size_t> &dims) override
	{
		m_components.front()->resizeInput(dims);
		for(size_t i = 1, j = components(); i < j; ++i)
		{
			m_components[i]->resizeInput(m_components[i - 1]->output().shape());
		}
	}
	
	/// Change the output dimensions of this module.
	virtual void resizeOutput(const Storage<size_t> &dims) override
	{
		m_components.back()->resizeOutput(dims);
		for(size_t i = components() - 1; i > 0; --i)
		{
			m_components[i - 1]->resizeOutput(m_components[i]->inBlame().shape());
		}
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		Tensor<T> *inp = const_cast<Tensor<T> *>(&input);
		for(Module<T> *component : m_components)
		{
			inp = &component->forward(*inp);
		}
		return *inp;
	}
	
	/// Backward propagate input and output blame, returning input blame.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outBlame) override
	{
		const Tensor<T> *blame = &outBlame;
		for(size_t i = components() - 1; i > 1; --i)
		{
			blame = &m_components[i]->backward(m_components[i - 1]->output(), *blame);
		}
		return m_components[0]->backward(input, *blame);
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_components.back()->output();
	}
	
	/// Cached input blame.
	virtual Tensor<T> &inBlame() override
	{
		return m_components.front()->inBlame();
	}
};

}

#endif
