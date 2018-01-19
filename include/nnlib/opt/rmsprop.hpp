#ifndef OPT_RMSPROP_HPP
#define OPT_RMSPROP_HPP

#include "optimizer.hpp"

namespace nnlib
{

template <typename T = NN_REAL_T>
class RMSProp : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
public:
    RMSProp(Module<T> &model, Critic<T> &critic);

    RMSProp &learningRate(T learningRate);
    T learningRate() const;

    virtual void reset() override;
    virtual RMSProp &step(const Tensor<T> &input, const Tensor<T> &target) override;

private:
    Tensor<T> &m_parameters;
    Tensor<T> &m_grads;
    Tensor<T> m_variance;
    T m_learningRate;
    T m_gamma;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::RMSProp<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/rmsprop.tpp"
#endif

#endif
