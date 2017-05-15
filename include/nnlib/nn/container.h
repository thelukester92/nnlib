#ifndef NN_CONTAINER_H
#define NN_CONTAINER_H

#include "module.h"

namespace nnlib
{

/// The abtract base class for neural network modules that are made up of sub-modules.
class Container : public Module
{
public:
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "container";
	}
	
	virtual ~Container()
	{
		for(Module *comp : m_components)
		{
			delete comp;
		}
	}
	
	/// Get a specific component from this container.
	Module *component(size_t index)
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
	Container &add(Module *component, Ms *...more)
	{
		add(component);
		add(more...);
		return *this;
	}
	
	/// Add a component to this container.
	virtual Container &add(Module *component)
	{
		m_components.push_back(component);
		return *this;
	}
	
	/// Remove and return a specific component from this container.
	virtual Module *remove(size_t index)
	{
		Module *comp = m_components[index];
		m_components.erase(index);
		return comp;
	}
	
	/// Set the batch size of this module.
	virtual Container &batch(size_t bats) override
	{
		for(Module *component : m_components)
		{
			component->batch(bats);
		}
		return *this;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters.
	virtual Storage<Tensor *> parameterList() override
	{
		Storage<Tensor *> params;
		for(Module *comp : m_components)
		{
			for(Tensor *param : comp->parameterList())
			{
				params.push_back(param);
			}
		}
		return params;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters' gradient.
	virtual Storage<Tensor *> gradList() override
	{
		Storage<Tensor *> blams;
		for(Module *comp : m_components)
		{
			for(Tensor *blam : comp->gradList())
			{
				blams.push_back(blam);
			}
		}
		return blams;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's internal state.
	virtual Storage<Tensor *> stateList() override
	{
		Storage<Tensor *> states;
		for(Module *comp : m_components)
		{
			for(Tensor *state : comp->stateList())
			{
				states.push_back(state);
			}
		}
		return states;
	}
protected:
	Storage<Module *> m_components;
};

}

#endif
