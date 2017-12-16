#ifndef NN_ELU_TPP
#define NN_ELU_TPP

#include "../elu.hpp"

namespace nnlib
{

template <typename T>
ELU<T>::ELU(T alpha) :
	m_alpha(alpha)
{
	NNAssertGreaterThanOrEquals(alpha, 0, "Expected positive alpha!");
}

template <typename T>
ELU<T>::ELU(const ELU<T> &module) :
	m_alpha(module.m_alpha)
{}

template <typename T>
ELU<T>::ELU(const Serialized &node) :
	m_alpha(node.get<T>("alpha"))
{}

template <typename T>
ELU<T> &ELU<T>::operator=(const ELU<T> &module)
{
	m_alpha = module.m_alpha;
	return *this;
}

template <typename T>
void ELU<T>::save(Serialized &node) const
{
	Map<T>::save(node);
	node.set("alpha", m_alpha);
}

template <typename T>
T ELU<T>::alpha() const
{
	return m_alpha;
}

template <typename T>
ELU<T> &ELU<T>::alpha(T alpha)
{
	NNAssertGreaterThanOrEquals(alpha, 0, "Expected positive alpha!");
	m_alpha = alpha;
	return *this;
}

template <typename T>
T ELU<T>::forwardOne(const T &x)
{
	return x > 0 ? x : (m_alpha * (exp(x) - 1));
}

template <typename T>
T ELU<T>::backwardOne(const T &x, const T &y)
{
	return x > 0 ? 1 : (y + m_alpha);
}

}

#endif
