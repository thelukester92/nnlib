#ifndef NN_CONCAT_TPP
#define NN_CONCAT_TPP

#include "../concat.hpp"

namespace nnlib
{

template <typename T>
Concat<T>::Concat(const Concat<T> &module) :
    Container<T>(static_cast<const Container<T> &>(module)),
    m_concatDim(module.m_concatDim)
{}

template <typename T>
Concat<T>::Concat(const Serialized &node) :
    Container<T>(node),
    m_concatDim(node.get<size_t>("concatDim"))
{}

template <typename T>
Concat<T> &Concat<T>::operator=(const Concat<T> &module)
{
    Container<T>::operator=(module);
    m_concatDim = module.m_concatDim;
    return *this;
}

template <typename T>
size_t Concat<T>::concatDim() const
{
    return m_concatDim;
}

template <typename T>
Concat<T> &Concat<T>::concatDim(size_t dim)
{
    m_concatDim = dim;
    return *this;
}

template <typename T>
void Concat<T>::save(Serialized &node) const
{
    Container<T>::save(node);
    node.set("concatDim", m_concatDim);
}

template <typename T>
Tensor<T> &Concat<T>::forward(const Tensor<T> &input)
{
    Storage<Tensor<T> *> outputs(components());
    for(size_t i = 0, count = components(); i < count; ++i)
        outputs[i] = &m_components[i]->forward(input);

    if(!m_output.sharedWith(outputs))
    {
        m_concatDim = std::min(m_concatDim, outputs[0]->dims() - 1);
        m_output = Tensor<T>::concatenate(outputs, m_concatDim);
    }

    return m_output;
}

template <typename T>
Tensor<T> &Concat<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    size_t offset = 0, stride;
    for(size_t i = 0, count = components(); i < count; ++i)
    {
        stride = m_components[i]->output().size(m_concatDim);
        m_components[i]->backward(input, outGrad.narrow(m_concatDim, offset, stride));
        offset += stride;
    }

    m_inGrad.resize(m_components[0]->inGrad().shape()).zeros();
    for(size_t i = 0, count = components(); i < count; ++i)
    {
        forEach([&](T x, T &y)
        {
            y += x;
        }, m_components[i]->inGrad(), m_inGrad);
    }

    return m_inGrad;
}

}

#endif
