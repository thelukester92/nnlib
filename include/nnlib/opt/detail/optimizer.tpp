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

template <typename T>
T Optimizer<T>::evaluate(const Tensor<T> &input, const Tensor<T> &target)
{
    return m_critic.forward(m_model.forward(input), target);
}

template <typename T>
void Optimizer<T>::reset()
{}

}

#endif
