#ifndef OPT_OPTIMIZER_HPP
#define OPT_OPTIMIZER_HPP

#include "../critics/critic.hpp"
#include "../nn/module.hpp"

namespace nnlib
{

/// \brief Base class for model optimizers.
///
/// This class owns the critic but not the model it is optimizing.
template <typename T = NN_REAL_T>
class Optimizer
{
public:
    Optimizer(Module<T> &model, Critic<T> *critic = nullptr);
    virtual ~Optimizer();

    Module<T> &model();
    Critic<T> &critic();
    Tensor<T> &params();
    Tensor<T> &grad();

    T learningRate() const;
    Optimizer<T> &learningRate(T learningRate);

    /// Evaluate the error on the given input/target pair.
    T evaluate(const Tensor<T> &input, const Tensor<T> &target);

    /// Reset the state of the optimizer (i.e. momentum).
    virtual void reset() = 0;

    /// Perform a single step of training given an input and a target.
    virtual Optimizer &step(const Tensor<T> &input, const Tensor<T> &target) = 0;

protected:
    Module<T> &m_model;
    Critic<T> *m_critic;
    Tensor<T> &m_params;
    Tensor<T> &m_grad;
    T m_learningRate;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Optimizer<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/optimizer.tpp"
#endif

#endif
