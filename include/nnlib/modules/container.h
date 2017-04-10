#ifndef CONTAINER_H
#define CONTAINER_H

#include "../module.h"

namespace nnlib
{

template <typename T = double>
class Container : public Module<T>
{
public:
	virtual ~Container()
	{
		for(Module<T> *component : m_components)
			delete component;
	}
	
	virtual void add(Module<T> *component)
	{
		m_components.push_back(component);
	}
	
	template <typename ... Ts>
	void add(Module<T> *component, Ts*...more)
	{
		add(component);
		add(more...);
	}
	
	size_t componentCount()
	{
		return m_components.size();
	}
	
	Module<T> *component(size_t i)
	{
		NNAssert(i < m_components.size(), "Invalid component index!");
		return m_components[i];
	}
	
	/// Remove the specified component. Caller is responsible for deleting it and fixing any problems caused by its removal.
	Module<T> *releaseComponent(size_t i)
	{
		NNAssert(i < m_components.size(), "Invalid component index!");
		Module<T> *component = m_components[i];
		m_components.erase(i);
		return component;
	}
protected:
	Vector<Module<T> *> m_components;
};

}

#endif
