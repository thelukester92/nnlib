#ifndef NN_CONVOLUTION_TPP
#define NN_CONVOLUTION_TPP

#include "../convolution.hpp"
#include "nnlib/math/algebra.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
Storage<size_t> Convolution<T>::calcInputShape(size_t kernelCount, size_t inChannels, const Storage<size_t> &kernelShape, const Storage<size_t> &stride, bool interleaved)
{
    NNAssertEquals(kernelShape.size(), 2, "Expected a 2D kernel shape!");
    NNAssertEquals(stride.size(), 2, "Expected a 2D stride!");
    size_t height = kernelShape[1] + stride[1];
    size_t width = kernelShape[0] + stride[0];
    if(interleaved)
        return { 1, height, width, inChannels };
    else
        return { 1, inChannels, height, width };
}

template <typename T>
Storage<size_t> Convolution<T>::calcOutputShape(size_t kernelCount, size_t inChannels, const Storage<size_t> &stride, const Storage<size_t> &pad, bool interleaved)
{
    NNAssertEquals(stride.size(), 2, "Expected a 2D stride!");
    NNAssertEquals(pad.size(), 2, "Expected a 2D pad!");
    size_t height = (pad[1] + 1) / stride[1] + 1;
    size_t width = (pad[0] + 1) / stride[0] + 1;
    if(interleaved)
        return { 1, height, width, kernelCount };
    else
        return { 1, kernelCount, height, width };
}

template <typename T>
Convolution<T>::Convolution(size_t kernelCount, size_t inChannels, const Storage<size_t> &kernelShape, const Storage<size_t> &stride, const Storage<size_t> &pad, bool interleaved) :
    Module<T>(
        calcInputShape(kernelCount, inChannels, kernelShape, stride, interleaved),
        calcOutputShape(kernelCount, inChannels, stride, pad, interleaved)
    ),
    m_kernels({ kernelCount, inChannels, kernelShape[0], kernelShape[1] }, true),
    m_kernelsGrad(m_kernels.shape(), true),
    m_bias(kernelCount),
    m_biasGrad(kernelCount),
    m_stride(stride),
    m_pad(pad),
    m_interleaved(interleaved)
{
    reset();
}

template <typename T>
Convolution<T>::Convolution(size_t kernelCount, size_t inChannels, size_t kernelShape, size_t stride, size_t pad, bool interleaved) :
    Convolution(kernelCount, inChannels, { kernelShape, kernelShape }, { stride, stride }, { pad, pad })
{}

template <typename T>
Convolution<T>::Convolution(const Convolution<T> &module) :
    Module<T>(module),
    m_kernels(module.m_kernels.copy()),
    m_kernelsGrad(m_kernels.shape(), true),
    m_bias(module.m_bias.copy()),
    m_biasGrad(m_bias.shape(), true),
    m_stride(module.m_stride),
    m_pad(module.m_pad),
    m_interleaved(module.m_interleaved)
{}

template <typename T>
Convolution<T>::Convolution(const Serialized &node) :
    Module<T>(node),
    m_kernels(node.get<Tensor<T>>("kernels")),
    m_kernelsGrad(m_kernels.shape(), true),
    m_bias(node.get<Tensor<T>>("bias")),
    m_biasGrad(m_bias.shape(), true),
    m_stride(node.get<Storage<size_t>>("stride")),
    m_pad(node.get<Storage<size_t>>("pad")),
    m_interleaved(node.get<bool>("interleaved"))
{}

template <typename T>
Convolution<T> &Convolution<T>::operator=(const Convolution<T> &module)
{
    Module<T>::operator=(module);
    if(this != &module)
    {
        m_kernels = module.m_kernels.copy();
        m_kernelsGrad.resize(m_kernels.shape());
        m_bias = module.m_bias.copy();
        m_biasGrad.resize(m_bias.shape());
        m_stride = module.m_stride;
        m_pad = module.m_pad;
        m_interleaved = module.m_interleaved;
    }
    return *this;
}

template <typename T>
size_t Convolution<T>::kernelCount() const
{
    return m_kernels.size(0);
}

template <typename T>
size_t Convolution<T>::inChannels() const
{
    return m_kernels.size(1);
}

template <typename T>
Storage<size_t> Convolution<T>::kernelShape() const
{
    return { m_kernels.size(2), m_kernels.size(3) };
}

template <typename T>
const Storage<size_t> &Convolution<T>::stride() const
{
    return m_stride;
}

template <typename T>
const Storage<size_t> &Convolution<T>::pad() const
{
    return m_pad;
}

template <typename T>
bool Convolution<T>::interleaved() const
{
    return m_interleaved;
}

template <typename T>
Tensor<T> Convolution<T>::kernels()
{
    return m_kernels;
}

template <typename T>
Tensor<T> Convolution<T>::bias()
{
    return m_bias;
}

template <typename T>
Convolution<T> &Convolution<T>::reset()
{
    T dev = 1.0 / sqrt(m_kernels.size(1) * m_kernels.size(2) * m_kernels.size(3));
    math::rand(m_kernels, -dev, dev);
    math::rand(m_bias, -dev, dev);
    return *this;
}

template <typename T>
void Convolution<T>::save(Serialized &node) const
{
    Module<T>::save(node);
    node.set("kernels", m_kernels);
    node.set("bias", m_bias);
    node.set("stride", m_stride);
    node.set("pad", m_pad);
    node.set("interleaved", m_interleaved);
}

template <typename T>
Tensor<T> &Convolution<T>::forward(const Tensor<T> &input)
{
    NNAssert(input.dims() == 3 || input.dims() == 4, "Expected 3D or 4D input!");

    Storage<size_t> shape;
    if(input.dims() == 3)
        shape = { 1, input.size(0), input.size(1), input.size(2) };
    else
        shape = input.shape();

    size_t inChannels = m_interleaved ? shape[3] : shape[1];
    size_t inHeight   = m_interleaved ? shape[1] : shape[2];
    size_t inWidth    = m_interleaved ? shape[2] : shape[3];

    NNAssertEquals(inChannels, m_kernels.size(1), "Incompatible input channels!");
    NNAssertGreaterThanOrEquals(inHeight, m_kernels.size(2) - m_pad[1] - m_stride[1] + 1, "Incopatible input height!");
    NNAssertGreaterThanOrEquals(inWidth, m_kernels.size(3) - m_pad[0] - m_stride[0] + 1, "Incopatible input width!");

    size_t outHeight  = (inHeight - m_kernels.size(2) + m_pad[1] + m_stride[1] - 1) / m_stride[1] + 1;
    size_t outWidth   = (inWidth - m_kernels.size(3) + m_pad[0] + m_stride[0] - 1) / m_stride[0] + 1;

    if(m_interleaved)
        m_output.resize(input.size(0), outWidth, outHeight, m_kernels.size(0));
    else
        m_output.resize(input.size(0), m_kernels.size(0), outWidth, outHeight);

    // todo: the actual convolution

    return m_output;
}

template <typename T>
Tensor<T> &Convolution<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    return m_inGrad;
}

template <typename T>
Storage<Tensor<T> *> Convolution<T>::paramsList()
{
    return { &m_kernels, &m_bias };
}

template <typename T>
Storage<Tensor<T> *> Convolution<T>::gradList()
{
    return { &m_kernelsGrad, &m_biasGrad };
}

}

#endif
