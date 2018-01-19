#ifndef OPT_RMSPROP_TPP
#define OPT_RMSPROP_TPP

#include "../rmsprop.hpp"

namespace nnlib
{

template <typename T>
RMSProp<T>::RMSProp(Module<T> &model, Critic<T> &critic) :
    Optimizer<T>(model, critic),
    m_parameters(model.params()),
    m_grads(model.grad()),
    m_learningRate(0.001),
    m_gamma(0.9)
{
    m_variance.resize(m_grads.size()).fill(0.0);
}

template <typename T>
RMSProp<T> &RMSProp<T>::learningRate(T learningRate)
{
    m_learningRate = learningRate;
    return *this;
}

template <typename T>
T RMSProp<T>::learningRate() const
{
    return m_learningRate;
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
    m_grads.fill(0);
    m_model.backward(input, m_critic.backward(m_model.forward(input), target));

    for(size_t i = 0, end = m_grads.size(); i != end; ++i)
    {
        // update variance
        m_variance(i) = m_gamma * m_variance(i) + (1 - m_gamma) * m_grads(i) * m_grads(i);

        // update parameters
        m_parameters(i) -= m_learningRate * m_grads(i) / (sqrt(m_variance(i)) + 1e-8);
    }

    return *this;
}

}

#endif
