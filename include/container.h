#ifndef CONTAINER_H
#define CONTAINER_H

#include "module.h"

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
protected:
	Vector<Module<T> *> m_components;
};

}

#endif
