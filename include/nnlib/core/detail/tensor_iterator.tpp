#ifndef CORE_TENSOR_ITERATOR_TPP
#define CORE_TENSOR_ITERATOR_TPP

#include "tensor_iterator.hpp"

namespace nnlib
{

template <typename T>
TensorIterator<T>::TensorIterator(const Tensor<TT> *tensor, bool end) :
    m_contiguous(tensor->contiguous()),
    m_shape(tensor->shape()),
    m_stride(tensor->strides()),
    m_indices(m_contiguous ? 1 : tensor->dims(), 0),
    m_ptr(const_cast<Tensor<TT> *>(tensor)->ptr())
{
    if(end || tensor->size() == 0)
    {
        m_indices[0] = m_shape[0];
        m_ptr += m_stride[0] * m_indices[0];
    }
}

template <typename T>
TensorIterator<T> &TensorIterator<T>::operator++()
{
    if(m_contiguous)
    {
        ++m_ptr;
        return *this;
    }

    size_t d = m_indices.size() - 1;
    ++m_indices[d];
    m_ptr += m_stride[d];

    while(m_indices[d] >= m_shape[d] && d > 0)
    {
        m_ptr -= m_stride[d] * m_indices[d];
        m_indices[d] = 0;

        --d;

        ++m_indices[d];
        m_ptr += m_stride[d];
    }

    return *this;
}

template <typename T>
TensorIterator<T> TensorIterator<T>::operator++(int)
{
    TensorIterator it = *this;
    ++(*this);
    return it;
}

template <typename T>
T &TensorIterator<T>::operator*()
{
    return *m_ptr;
}

template <typename T>
bool TensorIterator<T>::operator==(const TensorIterator<T> &other)
{
    return !(*this != other);
}

template <typename T>
bool TensorIterator<T>::operator!=(const TensorIterator<T> &other)
{
    if(m_contiguous)
        return m_ptr != other.m_ptr;
    else
        return m_indices != other.m_indices;
}

}

#endif
