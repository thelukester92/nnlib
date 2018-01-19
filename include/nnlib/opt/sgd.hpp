#ifndef OPT_SGD_HPP
#define OPT_SGD_HPP

#include "optimizer.hpp"

namespace nnlib
{

template <typename T = NN_REAL_T>
class SGD : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
public:
    SGD(Module<T> &model, Critic<T> &critic);

    SGD &learningRate(T learningRate);
    T learningRate() const;

    SGD &momentum(T momentum);
    T momentum() const;

    virtual void reset() override;
    virtual SGD &step(const Tensor<T> &input, const Tensor<T> &target) override;

private:
    Tensor<T> &m_parameters;
    Tensor<T> &m_grads;
    Tensor<T> m_velocity;
    T m_learningRate;
    T m_momentum;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::SGD<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/sgd.tpp"
#endif

#endif
