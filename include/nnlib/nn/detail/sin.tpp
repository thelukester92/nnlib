#ifndef NN_SIN_TPP
#define NN_SIN_TPP

#include "../sin.hpp"

namespace nnlib
{

template <typename T>
Sin<T>::Sin()
{}

template <typename T>
Sin<T>::Sin(const Serialized &)
{}

template <typename T>
Sin<T>::Sin(const Sin<T> &)
{}

template <typename T>
Sin<T> &Sin<T>::operator=(const Sin<T> &)
{
	return *this;
}

template <typename T>
T Sin<T>::forwardOne(const T &x)
{
	return sin(x);
}

template <typename T>
T Sin<T>::backwardOne(const T &x, const T &y)
{
	return cos(x);
}

}

#endif
