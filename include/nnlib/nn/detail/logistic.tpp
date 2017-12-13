#ifndef NN_LOGISTIC_TPP
#define NN_LOGISTIC_TPP

#include "../logistic.hpp"

namespace nnlib
{

template <typename T>
Logistic<T>::Logistic()
{}

template <typename T>
Logistic<T>::Logistic(const Serialized &)
{}

template <typename T>
Logistic<T>::Logistic(const Logistic<T> &)
{}

template <typename T>
Logistic<T> &Logistic<T>::operator=(const Logistic<T> &)
{
	return *this;
}

template <typename T>
T Logistic<T>::forwardOne(const T &x)
{
	return 1.0 / (1.0 + exp(-x));
}

template <typename T>
T Logistic<T>::backwardOne(const T &x, const T &y)
{
	return y * (1.0 - y);
}

}

#endif
