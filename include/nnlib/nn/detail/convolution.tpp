#ifndef NN_CONVOLUTION_TPP
#define NN_CONVOLUTION_TPP

#include "../convolution.hpp"
#include "nnlib/math/algebra.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
Convolution<T>::Convolution(size_t channels, size_t filters, size_t kWidth, size_t kHeight, size_t strideX, size_t strideY, bool pad, bool interleaved) :
    Module<T>(
        { 1, interleaved ? height : channels, interleaved ? width : height, interleaved ? channels : width },
        { 1, interleaved ? kHeight : filters, interleaved ? kWidth : kHeight, interleaved ? filters : kWidth }
    ),
    m_filters(filters, channels, kHeight, kWidth),
    m_filtersGrad(filters, channels, kHeight, kWidth),
    m_strideX(strideX),
    m_strideY(strideY),
    m_pad(pad),
    m_interleaved(interleaved)
{
    reset();
}

template <typename T>
Convolution<T>::Convolution(const Convolution<T> &module) :
    Convolution<T>(module),
    m_filters(module.m_filters.copy()),
    m_filtersGrad(m_filters.shape(), true),
    m_strideX(module.m_strideX),
    m_strideY(module.m_strideY),
    m_pad(module.m_pad),
    m_interleaved(module.m_interleaved)
{}

template <typename T>
Convolution<T>::Convolution(const Serialized &node) :
    Module<T>(node),
    m_filters(node.get<Tensor<T>>("filters")),
    m_filtersGrad(m_filters.shape(), true),
    m_strideX(node.get<size_t>("strideX")),
    m_strideY(node.get<size_t>("strideY")),
    m_pad(node.get<bool>("pad")),
    m_interleaved(node.get<bool>("interleaved"))
{}

template <typename T>
Convolution<T> &Convolution<T>::operator=(const Convolution<T> &module)
{
    Module<T>::operator=(module);
    if(this != &module)
    {
        m_filters = module.m_filters.copy();
        m_filtersGrad.resize(m_filters.shape());
        m_kWidth = module.m_kWidth;
        m_kHeight = module.m_kHeight;
        m_strideX = module.m_strideX;
        m_strideY = module.m_strideY;
        m_pad = module.m_pad;
        m_interleaved = module.m_interleaved;
    }
    return *this;
}

template <typename T>
size_t Convolution<T>::filters() const
{
    return m_filters.size(0);
}

template <typename T>
size_t Convolution<T>::kernelWidth() const
{
    return m_filters.size(2);
}

template <typename T>
size_t Convolution<T>::kernelHeight() const
{
    return m_filters.size(3);
}

template <typename T>
size_t Convolution<T>::strideX() const
{
    return m_strideX;
}

template <typename T>
size_t Convolution<T>::strideY() const
{
    return m_strideY;
}

template <typename T>
bool Convolution<T>::padded() const
{
    return m_pad;
}

template <typename T>
bool Convolution<T>::interleaved() const
{
    return m_interleaved;
}

template <typename T>
Convolution<T> &Convolution<T>::reset()
{
    T dev = 1.0 / sqrt(m_filters.size(1) * m_filters.size(2) * m_filters.size(3));
    math::rand(m_filters, -dev, dev);
    return *this;
}

template <typename T>
void Convolution<T>::save(Serialized &node) const
{
    Module<T>::save(node);
    node.set("filters", m_filters);
    node.set("strideX", m_strideX);
    node.set("strideY", m_strideY);
    node.set("pad", m_pad);
    node.set("interleaved", m_interleaved);
}

template <typename T>
Tensor<T> &Convolution<T>::forward(const Tensor<T> &input)
{
    return m_output;
}

template <typename T>
Tensor<T> &Linear<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    return m_inGrad;
}

template <typename T>
Storage<Tensor<T> *> Convolution<T>::paramsList()
{
    return { &m_filters };
}

template <typename T>
Storage<Tensor<T> *> Linear<T>::gradList()
{
    return { &m_filtersGrad };
}

}

#endif
