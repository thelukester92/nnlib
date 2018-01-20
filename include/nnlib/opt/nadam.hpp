#ifndef OPT_NADAM_HPP
#define OPT_NADAM_HPP

#include "optimizer.hpp"

namespace nnlib
{

template <typename T = NN_REAL_T>
class Nadam : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
using Optimizer<T>::m_params;
using Optimizer<T>::m_grad;
using Optimizer<T>::m_learningRate;
public:
    Nadam(Module<T> &model, Critic<T> *critic = nullptr);

    Nadam &beta1(T beta1);
    T beta1() const;

    Nadam &beta2(T beta2);
    T beta2() const;

    virtual void reset() override;
    virtual Nadam &step(const Tensor<T> &input, const Tensor<T> &target) override;

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
    extern template class nnlib::Nadam<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/nadam.tpp"
#endif

#endif
