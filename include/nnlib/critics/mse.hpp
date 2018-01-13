#ifndef CRITICS_MSE_HPP
#define CRITICS_MSE_HPP

#include "critic.hpp"

namespace nnlib
{

/// \brief Mean squared error critic.
///
/// When average = false, this is sum squared error.
template <typename T = NN_REAL_T>
class MSE : public Critic<T>
{
public:
    MSE(bool average = true);

    bool average() const;
    MSE &average(bool ave);

    /// L = 1/n sum_i( (input(i) - target(i))^2 )
    virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override;

    /// dL/di = 2/n (input(i) - target(i))
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override;
    
protected:
    using Critic<T>::m_inGrad;

private:
    bool m_average;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::MSE<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/mse.tpp"
#endif

#endif
