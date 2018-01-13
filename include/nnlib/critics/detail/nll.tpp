#ifndef CRTIICS_NLL_TPP
#define CRTIICS_NLL_TPP

#include "../nll.hpp"

namespace nnlib
{

template <typename T>
NLL<T>::NLL(bool average) :
    m_average(average)
{}

template <typename T>
bool NLL<T>::average() const
{
    return m_average;
}

template <typename T>
NLL<T> &NLL<T>::average(bool ave)
{
    m_average = ave;
    return *this;
}

template <typename T>
size_t NLL<T>::misclassifications(const Tensor<T> &input, const Tensor<T> &target)
{
    NNAssertEquals(input.size(0), target.size(0), "Incompatible operands!");
    NNAssertEquals(input.dims(), 2, "Expected matrix input!");
    NNAssertEquals(target.dims(), 2, "Expected matrix target!");
    NNAssertEquals(target.size(1), 1, "Expected single-column target!");

    size_t miss = 0;
    for(size_t i = 0, iend = input.size(0), jend = input.size(1); i < iend; ++i)
    {
        NNAssertGreaterThanOrEquals(target(i, 0), 0, "Expected positive target!");

        size_t max = 0;
        for(size_t j = 1; j < jend; ++j)
            if(input(i, j) > input(i, max))
                max = j;

        if(max != target(i, 0))
            ++miss;
    }

    return miss;
}

template <typename T>
T NLL<T>::forward(const Tensor<T> &input, const Tensor<T> &target)
{
    NNAssertEquals(input.size(0), target.size(0), "Incompatible operands!");
    NNAssertEquals(input.dims(), 2, "Expected matrix input!");
    NNAssertEquals(target.dims(), 2, "Expected matrix target!");
    NNAssertEquals(target.size(1), 1, "Expected single-column target!");

    T sum = 0;
    size_t j;
    for(size_t i = 0, iend = input.size(0); i < iend; ++i)
    {
        NNAssertGreaterThanOrEquals(target(i, 0), 0, "Expected positive target!");
        j = target(i, 0);
        sum -= input(i, j);
    }

    if(m_average)
        sum /= input.size();

    return sum;
}

template <typename T>
Tensor<T> &NLL<T>::backward(const Tensor<T> &input, const Tensor<T> &target)
{
    NNAssertEquals(input.size(0), target.size(0), "Incompatible operands!");
    NNAssertEquals(input.dims(), 2, "Expected matrix input!");
    NNAssertEquals(target.size(1), 1, "Expected single-column target!");

    m_inGrad.resize(input.shape()).fill(0);
    T weight = -1.0;

    if(m_average)
        weight /= input.size();

    size_t j;
    for(size_t i = 0, iend = input.size(0); i < iend; ++i)
    {
        NNAssertGreaterThanOrEquals(target(i, 0), 0, "Expected positive target!");
        j = target(i, 0);
        m_inGrad(i, j) = weight;
    }

    return m_inGrad;
}

}

#endif
