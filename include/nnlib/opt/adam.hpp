#ifndef OPT_ADAM_HPP
#define OPT_ADAM_HPP

#include "optimizer.hpp"

namespace nnlib
{

template <typename T = NN_REAL_T>
class Adam : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
using Optimizer<T>::m_params;
using Optimizer<T>::m_grad;
using Optimizer<T>::m_learningRate;
public:
    Adam(Module<T> &model, Critic<T> *critic = nullptr);

    Adam &beta1(T beta1);
    T beta1() const;

    Adam &beta2(T beta2);
    T beta2() const;

    virtual void reset() override;
    virtual Adam &step(const Tensor<T> &input, const Tensor<T> &target) override;

private:
    Tensor<T> m_mean;
    Tensor<T> m_variance;
    T m_beta1;
    T m_beta2;
    T m_normalize1;
    T m_normalize2;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Adam<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/adam.tpp"
#endif

#endif
