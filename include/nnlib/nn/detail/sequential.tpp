#ifndef NN_SEQUENTIAL_TPP
#define NN_SEQUENTIAL_TPP

#include "../sequential.hpp"

namespace nnlib
{

template <typename T>
Tensor<T> &Sequential<T>::forward(const Tensor<T> &input)
{
	Tensor<T> *inp = const_cast<Tensor<T> *>(&input);
	for(Module<T> *component : m_components)
		inp = &component->forward(*inp);
	return *inp;
}

template <typename T>
Tensor<T> &Sequential<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
	const Tensor<T> *grad = &outGrad;
	for(size_t i = components() - 1; i > 0; --i)
		grad = &m_components[i]->backward(m_components[i - 1]->output(), *grad);
	return m_components[0]->backward(input, *grad);
}

template <typename T>
Tensor<T> &Sequential<T>::output()
{
	return m_components.back()->output();
}

template <typename T>
Tensor<T> &Sequential<T>::inGrad()
{
	return m_components[0]->inGrad();
}

}

#endif
