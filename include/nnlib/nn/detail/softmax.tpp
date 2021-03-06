#ifndef NN_SOFT_MAX_TPP
#define NN_SOFT_MAX_TPP

#include "../softmax.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
SoftMax<T>::SoftMax() :
    Module<T>({ 1, 1 })
{}

template <typename T>
Tensor<T> &SoftMax<T>::forward(const Tensor<T> &input)
{
    NNAssertEquals(input.dims(), 2, "Expected matrix input!");
    m_output.resize(input.shape());

    for(size_t i = 0, iend = input.size(0); i < iend; ++i)
    {
        T max = math::max(input.narrow(0, i)), sum = 0;
        for(size_t j = 0, jend = input.size(1); j < jend; ++j)
            sum += (m_output(i, j) = exp(input(i, j) - max));
        for(size_t j = 0, jend = input.size(1); j < jend; ++j)
            m_output(i, j) /= sum;
    }

    return m_output;
}

template <typename T>
Tensor<T> &SoftMax<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    NNAssertEquals(input.dims(), 2, "Expected matrix input!");
    NNAssertEquals(input.shape(), m_output.shape(), "LogSoftMax::forward must be called first!");
    NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
    m_inGrad.resize(input.shape());

    for(size_t i = 0, iend = input.size(0); i < iend; ++i)
    {
        T sum = 0;
        for(size_t j = 0, jend = input.size(1); j < jend; ++j)
            sum += outGrad(i, j) * m_output(i, j);
        for(size_t j = 0, jend = input.size(1); j < jend; ++j)
            m_inGrad(i, j) = m_output(i, j) * (outGrad(i, j) - sum);
    }

    return m_inGrad;
}

}

#endif
