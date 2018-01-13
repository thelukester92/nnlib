#ifndef OPT_OPTIMIZER_TPP
#define OPT_OPTIMIZER_TPP

#include "../optimizer.hpp"

namespace nnlib
{

template <typename T>
Optimizer<T>::Optimizer(Module<T> &model, Critic<T> &critic) :
    m_model(model),
    m_critic(critic)
{}

template <typename T>
Optimizer<T>::~Optimizer()
{}

template <typename T>
Module<T> &Optimizer<T>::model()
{
    return m_model;
}

template <typename T>
Critic<T> &Optimizer<T>::critic()
{
    return m_critic;
}

}

#endif
