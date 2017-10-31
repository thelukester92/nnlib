#ifndef NN_CONTAINER_HPP
#define NN_CONTAINER_HPP

#include "module.hpp"

namespace nnlib
{

/// The abtract base class for neural network modules that are made up of sub-modules.
template <typename T = double>
class Container : public Module<T>
{
public:
	template <typename ... Ms>
	Container(Ms... components) :
		m_components({ static_cast<Module<T> *>(components)... })
	{}
	
	Container(const Container &module) :
		m_components(module.m_components)
	{
		for(Module<T> *&m : m_components)
		{
			/// \note intentionally not releasing; module still owns the original
			m = m->copy();
		}
	}
	
	Container(const Serialized &node) :
		m_components(node.get<Storage<Module<T> *>>("components"))
	{}
	
	Container &operator=(const Container &module)
	{
		Storage<Module<T> *> components = module.m_components;
		for(Module<T> *&m : components)
		{
			/// \note intentionally not releasing; module still owns the original
			m = m->copy();
		}
		
		for(Module<T> *m : m_components)
			delete m;
		
		m_components = components;
		
		return *this;
	}
	
	virtual ~Container()
	{
		for(Module<T> *comp : m_components)
			delete comp;
	}
	
	virtual void training(bool training = true) override
	{
		for(Module<T> *comp : m_components)
			comp->training(training);
	}
	
	virtual void forget() override
	{
		for(Module<T> *comp : m_components)
			comp->forget();
	}
	
	virtual void save(Serialized &node) const override
	{
		node.set("components", m_components);
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
	
	/// Remove and return a specific component from this container. Caller is responsible for deleting this module.
	virtual Module<T> *remove(size_t index)
	{
		Module<T> *comp = m_components[index];
		m_components.erase(index);
		return comp;
	}
	
	/// Remove all components from this container and delete them.
	virtual Container &clear()
	{
		for(Module<T> *comp : m_components)
			delete comp;
		m_components.clear();
		return *this;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters.
	virtual Storage<Tensor<T> *> paramsList() override
	{
		Storage<Tensor<T> *> params;
		for(Module<T> *comp : m_components)
			params.append(comp->paramsList());
		return params;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList() override
	{
		Storage<Tensor<T> *> blams;
		for(Module<T> *comp : m_components)
			blams.append(comp->gradList());
		return blams;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's internal state.
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states = Module<T>::stateList();
		for(Module<T> *comp : m_components)
			states.append(comp->stateList());
		return states;
	}
	
protected:
	Storage<Module<T> *> m_components;
};

}

NNRegisterType(Container, Module);

#endif
