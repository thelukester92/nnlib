#ifndef CONTAINER_H
#define CONTAINER_H

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
	
	/// Add a component to this container.
	virtual void add(Module<T> *component)
	{
		m_components.push_back(component);
	}
	
	/// Remove and return a specific component from this container.
	virtual Module<T> *remove(size_t index)
	{
		Module<T> *comp = m_components[index];
		m_components.erase(index);
		return comp;
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
	
	/// A vector of tensors filled with (views of) each sub-module's parameters' blame.
	virtual Storage<Tensor<T> *> blame() override
	{
		Storage<Tensor<T> *> blams;
		for(Module<T> *comp : m_components)
		{
			for(Tensor<T> *blam : comp->blame())
			{
				blams.push_back(blam);
			}
		}
		return blams;
	}
protected:
	Storage<Module<T> *> m_components;
};

}

#endif
