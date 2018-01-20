#ifndef OPT_OPTIMIZER_TPP
#define OPT_OPTIMIZER_TPP

#include "../optimizer.hpp"
#include "nnlib/critics/mse.hpp"

namespace nnlib
{

template <typename T>
Optimizer<T>::Optimizer(Module<T> &model, Critic<T> *critic) :
    m_model(model),
    m_critic(critic != nullptr ? critic : new MSE<T>()),
    m_params(model.params()),
    m_grad(model.grad()),
    m_learningRate(0.01)
{}

template <typename T>
Optimizer<T>::~Optimizer()
{
    delete m_critic;
}

template <typename T>
Module<T> &Optimizer<T>::model()
{
    return m_model;
}

template <typename T>
Critic<T> &Optimizer<T>::critic()
{
    return *m_critic;
}

template <typename T>
Tensor<T> &Optimizer<T>::params()
{
    return m_params;
}

template <typename T>
Tensor<T> &Optimizer<T>::grad()
{
    return m_grad;
}

template <typename T>
T Optimizer<T>::learningRate() const
{
    return m_learningRate;
}

template <typename T>
Optimizer<T> &Optimizer<T>::learningRate(T learningRate)
{
    m_learningRate = learningRate;
    return *this;
}

template <typename T>
T Optimizer<T>::evaluate(const Tensor<T> &input, const Tensor<T> &target)
{
    return m_critic->forward(m_model.forward(input), target);
}

}

#endif
