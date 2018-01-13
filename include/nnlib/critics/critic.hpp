#ifndef CRITICS_CRITIC_HPP
#define CRITICS_CRITIC_HPP

#include "../core/tensor.hpp"

namespace nnlib
{

template <typename T>
class Tensor;

template <typename T = NN_REAL_T>
class Critic
{
public:
    virtual ~Critic();

    /// Calculate the loss (how far input is from target).
    virtual T forward(const Tensor<T> &input, const Tensor<T> &target) = 0;

    /// Calculate the gradient of the loss w.r.t. the input.
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) = 0;

    /// Get cached input gradient.
    Tensor<T> &inGrad();
    const Tensor<T> &inGrad() const;

protected:
    Tensor<T> m_inGrad;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Critic<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/critic.tpp"
#endif

#endif
