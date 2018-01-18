#ifndef NN_RELU_TPP
#define NN_RELU_TPP

#include "../relu.hpp"
#include <math.h>

namespace nnlib
{

template <typename T>
ReLU<T>::ReLU(T leak) :
    m_leak(leak)
{
    NNAssertGreaterThanOrEquals(leak, 0, "Expected positive leak!");
    NNAssertLessThan(leak, 1, "Expected leak to be a percentage!");
}

template <typename T>
ReLU<T>::ReLU(const ReLU<T> &module) :
    Map<T>(module),
    m_leak(module.m_leak)
{}

template <typename T>
ReLU<T>::ReLU(const Serialized &node) :
    Map<T>(node),
    m_leak(node.get<T>("leak"))
{}

template <typename T>
ReLU<T> &ReLU<T>::operator=(const ReLU<T> &module)
{
    Map<T>::operator=(module);
    m_leak = module.m_leak;
    return *this;
}

template <typename T>
void ReLU<T>::save(Serialized &node) const
{
    Map<T>::save(node);
    node.set("leak", m_leak);
}

template <typename T>
T ReLU<T>::leak() const
{
    return m_leak;
}

template <typename T>
ReLU<T> &ReLU<T>::leak(T leak)
{
    NNAssertGreaterThanOrEquals(leak, 0, "Expected positive leak!");
    NNAssertLessThan(leak, 1, "Expected leak to be a percentage!");
    m_leak = leak;
    return *this;
}

template <typename T>
T ReLU<T>::forwardOne(const T &x)
{
    return x > 0 ? x : m_leak * x;
}

template <typename T>
T ReLU<T>::backwardOne(const T &x, const T &y)
{
    return x > 0 ? 1 : m_leak;
}

}

#endif
