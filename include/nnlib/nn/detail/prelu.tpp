#ifndef NN_PRELU_TPP
#define NN_PRELU_TPP

#include "../prelu.hpp"
#include "nnlib/math/math.hpp"
#include <math.h>

namespace nnlib
{

template <typename T>
PReLU<T>::PReLU(size_t size) :
    Module<T>({ 1, size }),
    m_leaks(math::rand(Tensor<T>(size), 0, 1)),
    m_grads(size)
{}

template <typename T>
PReLU<T>::PReLU(const PReLU<T> &module) :
    Module<T>(module),
    m_leaks(module.m_leaks.copy()),
    m_grads(m_leaks.shape(), true)
{}

template <typename T>
PReLU<T>::PReLU(const Serialized &node) :
    Module<T>(node),
    m_leaks(node.get<Tensor<T>>("leaks")),
    m_grads(m_leaks.shape(), true)
{}

template <typename T>
PReLU<T> &PReLU<T>::operator=(const PReLU<T> &module)
{
    Module<T>::operator=(module);
    m_leaks = module.m_leaks.copy();
    m_grads.resize(m_leaks.shape());
    return *this;
}

template <typename T>
void PReLU<T>::save(Serialized &node) const
{
    Map<T>::save(node);
    node.set("leaks", m_leaks);
}

template <typename T>
T PReLU<T>::leak(size_t i) const
{
    return m_leaks(i);
}

template <typename T>
Tensor<T> &PReLU<T>::forward(const Tensor<T> &input)
{
    NNAssert(input.dims() == 1 || input.dims() == 2, "Expected vector or matrix input!");
    m_output.resize(input.shape());
    math::clip(m_leaks, 0, 1);

    if(input.dims() == 1)
    {
        NNAssertEquals(input.size(), m_leaks.size(), "Incompatible input!");
        for(size_t i = 0, n = input.size(); i < n; ++i)
            m_output(i) = input(i) > 0 ? input(i) : m_leaks(i) * input(i);
    }
    else
    {
        NNAssertEquals(input.size(1), m_leaks.size(), "Incompatible input!");
        for(size_t i = 0, n = input.size(0), m = input.size(1); i < n; ++i)
            for(size_t j = 0; j < m; ++j)
                m_output(i, j) = input(i, j) > 0 ? input(i, j) : m_leaks(j) * input(i, j);
    }

    return m_output;
}

template <typename T>
Tensor<T> &PReLU<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    NNAssert(input.dims() == 1 || input.dims() == 2, "Expected vector or matrix input!");
    NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
    m_inGrad.resize(input.shape());
    math::clip(m_leaks, 0, 1);

    if(input.dims() == 1)
    {
        NNAssertEquals(input.size(), m_leaks.size(), "Incompatible input!");
        for(size_t i = 0, n = input.size(); i < n; ++i)
        {
            m_inGrad(i) = outGrad(i) * (input(i) > 0 ? 1 : m_leaks(i));
            m_grads(i) += outGrad(i) * (input(i) > 0 ? 0 : input(i));
        }
    }
    else
    {
        NNAssertEquals(input.size(1), m_leaks.size(), "Incompatible input!");
        for(size_t i = 0, n = input.size(0), m = input.size(1); i < n; ++i)
        {
            for(size_t j = 0; j < m; ++j)
            {
                m_inGrad(i, j) = outGrad(i, j) * (input(i, j) > 0 ? 1 : m_leaks(j));
                m_grads(j) += outGrad(i, j) * (input(i, j) > 0 ? 0 : input(i, j));
            }
        }
    }

    return m_inGrad;
}

template <typename T>
Storage<Tensor<T> *> PReLU<T>::paramsList()
{
    return { &m_leaks };
}

template <typename T>
Storage<Tensor<T> *> PReLU<T>::gradList()
{
    return { &m_grads };
}

}

#endif
