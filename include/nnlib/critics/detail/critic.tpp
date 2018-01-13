#ifndef CRITICS_CRITIC_TPP
#define CRITICS_CRITIC_TPP

#include "../critic.hpp"

namespace nnlib
{

template <typename T>
Critic<T>::~Critic()
{}

template <typename T>
Tensor<T> &Critic<T>::inGrad()
{
    return m_inGrad;
}

template <typename T>
const Tensor<T> &Critic<T>::inGrad() const
{
    return m_inGrad;
}

}

#endif
