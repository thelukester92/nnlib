#ifndef OPT_RMSPROP_TPP
#define OPT_RMSPROP_TPP

#include "../rmsprop.hpp"

namespace nnlib
{

template <typename T>
RMSProp<T>::RMSProp(Module<T> &model, Critic<T> *critic) :
    Optimizer<T>(model, critic),
    m_gamma(0.9)
{
    m_variance.resize(m_grad.size()).fill(0.0);
}

template <typename T>
T RMSProp<T>::gamma() const
{
    return m_gamma;
}

template <typename T>
RMSProp<T> &RMSProp<T>::gamma(T gamma)
{
    m_gamma = gamma;
    return *this;
}

template <typename T>
void RMSProp<T>::reset()
{
    m_variance.fill(0);
}

template <typename T>
RMSProp<T> &RMSProp<T>::step(const Tensor<T> &input, const Tensor<T> &target)
{
    // calculate gradient
    m_grad.fill(0);
    m_model.backward(input, m_critic->backward(m_model.forward(input), target));

    for(size_t i = 0, end = m_grad.size(); i != end; ++i)
    {
        // update variance
        m_variance(i) = m_gamma * m_variance(i) + (1 - m_gamma) * m_grad(i) * m_grad(i);

        // update parameters
        m_params(i) -= m_learningRate * m_grad(i) / (sqrt(m_variance(i)) + 1e-8);
    }

    return *this;
}

}

#endif
