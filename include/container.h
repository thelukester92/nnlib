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
		m_components.push_back(&component);
	}
protected:
	Vector<Module<T> *> m_components;
};

}

#endif
