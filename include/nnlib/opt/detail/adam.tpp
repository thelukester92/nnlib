#ifndef OPT_ADAM_TPP
#define OPT_ADAM_TPP

#include "../adam.hpp"
#include "nnlib/math/math.hpp"

namespace nnlib
{

template <typename T>
Adam<T>::Adam(Module<T> &model, Critic<T> *critic) :
    Optimizer<T>(model, critic),
    m_beta1(0.9),
    m_beta2(0.999),
    m_normalize1(1),
    m_normalize2(1)
{
    math::fill(m_mean.resize(m_grad.size()), 0);
    math::fill(m_variance.resize(m_grad.size()), 0);
}

template <typename T>
Adam<T> &Adam<T>::beta1(T beta1)
{
    m_beta1 = beta1;
    return *this;
}

template <typename T>
T Adam<T>::beta1() const
{
    return m_beta1;
}

template <typename T>
Adam<T> &Adam<T>::beta2(T beta2)
{
    m_beta2 = beta2;
    return *this;
}

template <typename T>
T Adam<T>::beta2() const
{
    return m_beta2;
}

template <typename T>
void Adam<T>::reset()
{
    m_normalize1 = 1;
    m_normalize2 = 1;
    math::fill(m_mean, 0);
    math::fill(m_variance, 0);
}

template <typename T>
Adam<T> &Adam<T>::step(const Tensor<T> &input, const Tensor<T> &target)
{
    m_normalize1 *= m_beta1;
    m_normalize2 *= m_beta2;

    T lr = m_learningRate / (1 - m_normalize1) * sqrt(1 - m_normalize2);

    // calculate gradient
    math::fill(m_grad, 0);
    m_model.backward(input, m_critic->backward(m_model.forward(input), target));

    for(size_t i = 0, end = m_grad.size(); i != end; ++i)
    {
        // update mean
        m_mean(i) = m_beta1 * m_mean(i) + (1 - m_beta1) * m_grad(i);

        // update variance
        m_variance(i) = m_beta2 * m_variance(i) + (1 - m_beta2) * m_grad(i) * m_grad(i);

        // update parameters
        m_params(i) -= lr * m_mean(i) / (sqrt(m_variance(i)) + 1e-8);
    }

    return *this;
}

}

#endif
