#ifndef OPT_NADAM_TPP
#define OPT_NADAM_TPP

#include "../nadam.hpp"

namespace nnlib
{

template <typename T>
Nadam<T>::Nadam(Module<T> &model, Critic<T> &critic) :
    Optimizer<T>(model, critic),
    m_parameters(model.params()),
    m_grads(model.grad()),
    m_learningRate(0.001),
    m_beta1(0.9),
    m_beta2(0.999),
    m_normalize1(1),
    m_normalize2(1)
{
    m_mean.resize(m_grads.size()).fill(0);
    m_variance.resize(m_grads.size()).fill(0);
}

template <typename T>
Nadam<T> &Nadam<T>::learningRate(T learningRate)
{
    m_learningRate = learningRate;
    return *this;
}

template <typename T>
T Nadam<T>::learningRate() const
{
    return m_learningRate;
}

template <typename T>
Nadam<T> &Nadam<T>::beta1(T beta1)
{
    m_beta1 = beta1;
    return *this;
}

template <typename T>
T Nadam<T>::beta1() const
{
    return m_beta1;
}

template <typename T>
Nadam<T> &Nadam<T>::beta2(T beta2)
{
    m_beta2 = beta2;
    return *this;
}

template <typename T>
T Nadam<T>::beta2() const
{
    return m_beta2;
}

template <typename T>
void Nadam<T>::reset()
{
    m_normalize1 = 1;
    m_normalize2 = 1;
    m_mean.fill(0);
    m_variance.fill(0);
}

template <typename T>
Nadam<T> &Nadam<T>::step(const Tensor<T> &input, const Tensor<T> &target)
{
    m_normalize1 *= m_beta1;
    m_normalize2 *= m_beta2;

    T lr = m_learningRate / (1 - m_normalize1) * sqrt(1 - m_normalize2);

    // calculate gradient
    m_grads.fill(0);
    m_model.backward(input, m_critic.backward(m_model.forward(input), target));

    for(size_t i = 0, end = m_grads.size(); i != end; ++i)
    {
        // update mean
        m_mean(i) = m_beta1 * m_mean(i) + (1 - m_beta1) * m_grads(i);

        // update variance
        m_variance(i) = m_beta2 * m_variance(i) + (1 - m_beta2) * m_grads(i) * m_grads(i);

        // update parameters
        m_parameters(i) -= lr * ((1 - m_beta1) * m_grads(i) + m_beta1 * m_mean(i)) / (sqrt(m_variance(i)) + 1e-8);
    }

    return *this;
}

}

#endif
