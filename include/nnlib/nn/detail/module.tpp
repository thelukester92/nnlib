#ifndef NN_MODULE_TPP
#define NN_MODULE_TPP

#include "../module.hpp"

namespace nnlib
{

template <typename T>
Module<T>::Module()
{}

template <typename T>
Module<T>::~Module()
{}

template <typename T>
Module<T> *Module<T>::copy() const
{
	return Factory<Module<T>>::constructCopy(this);
}

template <typename T>
void Module<T>::training(bool training)
{}

template <typename T>
void Module<T>::forget()
{
	state().fill(0);
}

template <typename T>
Storage<Tensor<T> *> Module<T>::paramsList()
{
	return {};
}

template <typename T>
Storage<Tensor<T> *> Module<T>::gradList()
{
	return {};
}

template <typename T>
Storage<Tensor<T> *> Module<T>::stateList()
{
	return { &m_output };
}

template <typename T>
Tensor<T> &Module<T>::params()
{
	auto list = paramsList();
	if(!m_params.sharedWith(list))
		m_params = Tensor<T>::vectorize(list);
	return m_params;
}

template <typename T>
Tensor<T> &Module<T>::grad()
{
	auto list = gradList();
	if(!m_grad.sharedWith(list))
		m_grad = Tensor<T>::vectorize(list);
	return m_grad;
}

template <typename T>
Tensor<T> &Module<T>::state()
{
	auto list = stateList();
	if(!m_state.sharedWith(list))
		m_state = Tensor<T>::vectorize(list);
	return m_state;
}

template <typename T>
Tensor<T> &Module<T>::output()
{
	return m_output;
}

template <typename T>
const Tensor<T> &Module<T>::output() const
{
	return const_cast<Module<T> *>(this)->output();
}

template <typename T>
Tensor<T> &Module<T>::inGrad()
{
	return m_inGrad;
}

template <typename T>
const Tensor<T> &Module<T>::inGrad() const
{
	return const_cast<Module<T> *>(this)->inGrad();
}

}

#endif
