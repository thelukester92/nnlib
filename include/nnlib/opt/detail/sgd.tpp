#ifndef OPT_SGD_TPP
#define OPT_SGD_TPP

#include "../sgd.hpp"
#include "nnlib/math/algebra.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
SGD<T>::SGD(Module<T> &model, Critic<T> *critic) :
    Optimizer<T>(model, critic),
    m_velocity(m_grad.size()),
    m_momentum(0)
{
    math::fill(m_velocity, 0);
}

template <typename T>
SGD<T> &SGD<T>::momentum(T momentum)
{
    m_momentum = momentum;
    return *this;
}

template <typename T>
T SGD<T>::momentum() const
{
    return m_momentum;
}

template <typename T>
void SGD<T>::reset()
{
    math::fill(m_velocity, 0);
}

template <typename T>
SGD<T> &SGD<T>::step(const Tensor<T> &input, const Tensor<T> &target)
{
    // calculate gradient
    math::fill(m_grad, 0);
    m_model.backward(input, m_critic->backward(m_model.forward(input), target));

    // Nesterov momentum
    if(m_momentum)
    {
        math::scale(m_velocity, m_momentum);
        math::vAdd_v(m_grad, m_velocity, -m_learningRate);
        math::vAdd_v(m_velocity, m_params, m_momentum);
    }

    // update parameters
    math::vAdd_v(m_grad, m_params, -m_learningRate);

    return *this;
}

}

#endif
