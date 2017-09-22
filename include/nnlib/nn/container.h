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
	using Module<T>::training;
	
	Container() {}
	
	Container(const Container &module) : m_components(module.m_components)
	{
		for(Module<T> *&m : m_components)
			m = m->copy(); /// \note intentionally not releasing; module still owns the original
	}
	
	Container &operator=(const Container &module)
	{
		m_components = module.m_components;
		for(Module<T> *&m : m_components)
			m = m->copy(); /// \note intentionally not releasing; module still owns the original
		return *this;
	}
	
	virtual ~Container()
	{
		for(Module<T> *comp : m_components)
			delete comp;
	}
	
	/// Sets whether this module is in training mode.
	virtual Container &training(bool training) override
	{
		Module<T>::training(training);
		for(Module<T> *comp : m_components)
			comp->training(training);
		return *this;
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
	
	/// Remove all components from this container.
	virtual Container &clear()
	{
		for(Module<T> *comp : m_components)
			delete comp;
		m_components.clear();
		return *this;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters.
	virtual Storage<Tensor<T> *> parameterList() override
	{
		Storage<Tensor<T> *> params;
		for(Module<T> *comp : m_components)
			for(Tensor<T> *param : comp->parameterList())
				params.push_back(param);
		return params;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList() override
	{
		Storage<Tensor<T> *> blams;
		for(Module<T> *comp : m_components)
			for(Tensor<T> *blam : comp->gradList())
				blams.push_back(blam);
		return blams;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's internal state.
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states;
		for(Module<T> *comp : m_components)
			for(Tensor<T> *state : comp->stateList())
				states.push_back(state);
		return states;
	}
	
	/// Reset the internal state of this module.
	virtual Container &forget() override
	{
		for(Module<T> *comp : m_components)
			comp->forget();
		return *this;
	}
	
protected:
	Storage<Module<T> *> m_components;
};

}

NNRegisterType(Container, Module);

#endif
