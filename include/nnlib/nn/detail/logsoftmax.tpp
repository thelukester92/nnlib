#ifndef NN_LOG_SOFT_MAX_TPP
#define NN_LOG_SOFT_MAX_TPP

#include "../logsoftmax.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
LogSoftMax<T>::LogSoftMax()
{}

template <typename T>
LogSoftMax<T>::LogSoftMax(const LogSoftMax &module)
{}

template <typename T>
LogSoftMax<T>::LogSoftMax(const Serialized &node)
{}

template <typename T>
LogSoftMax<T> &LogSoftMax<T>::operator=(const LogSoftMax &module)
{
    return *this;
}

template <typename T>
void LogSoftMax<T>::save(Serialized &node) const
{}

template <typename T>
Tensor<T> &LogSoftMax<T>::forward(const Tensor<T> &input)
{
    NNAssertEquals(input.dims(), 2, "Expected matrix input!");
    m_output.resize(input.shape());

    for(size_t i = 0, iend = input.size(0); i < iend; ++i)
    {
        T max = math::max(input.narrow(0, i)), sum = 0;
        for(size_t j = 0, jend = input.size(1); j < jend; ++j)
            sum += exp(input(i, j) - max);
        sum = max + log(sum);
        for(size_t j = 0, jend = input.size(1); j < jend; ++j)
            m_output(i, j) = input(i, j) - sum;
    }

    return m_output;
}

template <typename T>
Tensor<T> &LogSoftMax<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    NNAssertEquals(input.dims(), 2, "Expected matrix input!");
    NNAssertEquals(input.shape(), m_output.shape(), "LogSoftMax::forward must be called first!");
    NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
    m_inGrad.resize(input.shape());

    for(size_t i = 0, iend = input.size(0); i < iend; ++i)
    {
        T sum = math::sum(outGrad.narrow(0, i));
        for(size_t j = 0, jend = input.size(1); j < jend; ++j)
            m_inGrad(i, j) = outGrad(i, j) - exp(m_output(i, j)) * sum;
    }

    return m_inGrad;
}

}

#endif
