#ifndef NN_IDENTITY_TPP
#define NN_IDENTITY_TPP

#include "../identity.hpp"

namespace nnlib
{

template <typename T>
Identity<T>::Identity()
{}

template <typename T>
Identity<T>::Identity(const Serialized &)
{}

template <typename T>
Identity<T>::Identity(const Identity<T> &)
{}

template <typename T>
Identity<T> &Identity<T>::operator=(const Identity<T> &)
{
    return *this;
}

template <typename T>
void Identity<T>::save(Serialized &node) const
{}

template <typename T>
Tensor<T> &Identity<T>::forward(const Tensor<T> &input)
{
    return m_output.resize(input.shape()).copy(input);
}

template <typename T>
Tensor<T> &Identity<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
    NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
    return m_inGrad.resize(outGrad.shape()).copy(outGrad);
}

}

#endif
