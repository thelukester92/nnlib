#ifndef NN_CONTAINER_H
#define NN_CONTAINER_H

#include "module.h"

namespace nnlib
{

/// The abtract base class for neural network modules that are made up of sub-modules.
template <typename T = double>
class Container : public Module<T>
{
public:
	virtual ~Container()
	{
		for(Module<T> *comp : m_components)
		{
			delete comp;
		}
	}
	
	/// Get a specific component from this container.
	Module<T> *component(size_t index)
	{
		return m_components[index];
	}
	
	/// Get the number of components in this container.
	size_t components() const
	{
		return m_components.size();
	}
	
	/// Add multiple components to this container.
	template <typename ... Ms>
	Container &add(Module<T> *component, Ms *...more)
	{
		add(component);
		add(more...);
		return *this;
	}
	
	/// Add a component to this container.
	virtual Container &add(Module<T> *component)
	{
		m_components.push_back(component);
		return *this;
	}
	
	/// Remove and return a specific component from this container.
	virtual Module<T> *remove(size_t index)
	{
		Module<T> *comp = m_components[index];
		m_components.erase(index);
		return comp;
	}
	
	/// Set the batch size of this module.
	virtual Container &batch(size_t bats) override
	{
		for(Module<T> *component : m_components)
		{
			component->batch(bats);
		}
		return *this;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters.
	virtual Storage<Tensor<T> *> parameters() override
	{
		Storage<Tensor<T> *> params;
		for(Module<T> *comp : m_components)
		{
			for(Tensor<T> *param : comp->parameters())
			{
				params.push_back(param);
			}
		}
		return params;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters' gradient.
	virtual Storage<Tensor<T> *> grad() override
	{
		Storage<Tensor<T> *> blams;
		for(Module<T> *comp : m_components)
		{
			for(Tensor<T> *blam : comp->grad())
			{
				blams.push_back(blam);
			}
		}
		return blams;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's internal state.
	virtual Storage<Tensor<T> *> innerState() override
	{
		Storage<Tensor<T> *> states;
		for(Module<T> *comp : m_components)
		{
			for(Tensor<T> *state : comp->innerState())
			{
				states.push_back(state);
			}
		}
		return states;
	}
protected:
	Storage<Module<T> *> m_components;
};

}

#endif
