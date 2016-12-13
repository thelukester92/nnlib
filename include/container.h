#ifndef CONTAINER_H
#define CONTAINER_H

#include "module.h"

namespace nnlib
{

template <typename T>
class Container : public Module<T>
{
public:
	void add(Module<T> &component)
	{
		m_components.resize(m_components.size() + 1);
		m_components[m_components.size() - 1] = &component;
	}
protected:
	Vector<Module<T> *> m_components;
};

}

#endif
