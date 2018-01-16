#ifndef NN_LINEAR_TPP
#define NN_LINEAR_TPP

#include "../linear.hpp"
#include "nnlib/math/algebra.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
Linear<T>::Linear(size_t inps, size_t outs, bool bias) :
    m_weights(inps, outs),
    m_weightsGrad(inps, outs),
    m_useBias(bias),
    m_bias(bias ? outs : 0),
    m_biasGrad(bias ? outs : 0)
{
    reset();
}

template <typename T>
Linear<T>::Linear(const Linear<T> &module) :
    m_weights(module.m_weights.copy()),
    m_weightsGrad(m_weights.shape(), true),
    m_useBias(module.m_useBias),
    m_bias(module.m_bias.copy()),
    m_biasGrad(m_bias.shape(), true)
{}

template <typename T>
Linear<T>::Linear(const Serialized &node) :
    m_weights(node.get<Tensor<T>>("weights")),
    m_weightsGrad(m_weights.shape(), true),
    m_useBias(node.get<bool>("useBias")),
    m_bias(node.get<Tensor<T>>("bias")),
    m_biasGrad(m_bias.shape(), true)
{
    NNAssertEquals(m_weights.dims(), 2, "Expected matrix weights!");
    NNAssert(!m_useBias || m_bias.dims() == 1, "Expected vector bias!");
    NNAssert(!m_useBias || m_weights.size(1) == m_bias.size(), "Incompatible weights and bias!");
}

template <typename T>
Linear<T> &Linear<T>::operator=(Linear<T> module)
{
    swap(*this, module);
    return *this;
}

template <typename T>
void swap(Linear<T> &a, Linear<T> &b)
{
    using std::swap;
    swap(a.m_weights, b.m_weights);
    swap(a.m_weightsGrad, b.m_weightsGrad);
    swap(a.m_useBias, b.m_useBias);
    swap(a.m_bias, b.m_bias);
    swap(a.m_biasGrad, b.m_biasGrad);
}

template <typename T>
bool Linear<T>::biased() const
{
    return m_useBias;
}

template <typename T>
Linear<T> &Linear<T>::reset()
{
    T dev = 1.0 / sqrt(m_weights.size(1));
    math::rand(m_weights, -dev, dev);

    if(m_useBias)
        math::rand(m_bias, -dev, dev);

    return *this;
}

template <typename T>
Tensor<T> Linear<T>::weights()
{
    return m_weights;
}

template <typename T>
Tensor<T> Linear<T>::bias()
{
    NNHardAssert(m_useBias, "This is an unbiased module!");
    return m_bias;
}

template <typename T>
void Linear<T>::save(Serialized &node) const
{
    node.set("weights", m_weights);
    node.set("useBias", m_useBias);
    node.set("bias", m_bias);
}

template <typename T>
Tensor<T> &Linear<T>::forward(const Tensor<T> &input)
{
    if(input.dims() == 1)
    {
        m_output.resize(m_weights.size(1));
        if(m_useBias)
            Algebra<T>::vAdd_mtv(m_weights, input, m_output.copy(m_bias));
        else
            Algebra<T>::vAdd_mtv(m_weights, input, m_output, 1, 0);
    }
    else if(input.dims() == 2)
    {
        m_output.resize(input.size(0), m_weights.size(1));
        Algebra<T>::mAdd_mm(input, m_weights, m_output, 1, 0);
        if(m_useBias)
            Algebra<T>::mAdd_vv(m_ones.resize(input.size(0)).fill(1), m_bias, m_output);
    }
    else
    {
        throw Error("Expected vector or matrix input!");
    }

    return m_output;
}

template <typename T>
Tensor<T> &Linear<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    NNAssertEquals(input.dims(), outGrad.dims(), "Incompatible input and outGrad!");
    if(input.dims() == 1)
    {
        Algebra<T>::mAdd_vv(input, outGrad, m_weightsGrad);
        if(m_useBias)
            Algebra<T>::vAdd_v(outGrad, m_biasGrad);

        m_inGrad.resize(m_weights.size(0));
        Algebra<T>::vAdd_mv(m_weights, outGrad, m_inGrad, 1, 0);
    }
    else if(input.dims() == 2)
    {
        Algebra<T>::mAdd_mtm(input, outGrad, m_weightsGrad);
        if(m_useBias)
            Algebra<T>::vAdd_mtv(outGrad, m_ones.resize(input.size(0)).fill(1), m_biasGrad);

        m_inGrad.resize(input.size(0), m_weights.size(0));
        Algebra<T>::mAdd_mmt(outGrad, m_weights, m_inGrad, 1, 0);
    }
    else
    {
        throw Error("Expected vector or matrix input and outGrad!");
    }

    return m_inGrad;
}

template <typename T>
Storage<Tensor<T> *> Linear<T>::paramsList()
{
    if(m_useBias)
        return { &m_weights, &m_bias };
    else
        return { &m_weights };
}

template <typename T>
Storage<Tensor<T> *> Linear<T>::gradList()
{
    if(m_useBias)
        return { &m_weightsGrad, &m_biasGrad };
    else
        return { &m_weightsGrad };
}

}

#endif
