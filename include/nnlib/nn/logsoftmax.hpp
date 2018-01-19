#ifndef NN_LOG_SOFT_MAX_HPP
#define NN_LOG_SOFT_MAX_HPP

#include "module.hpp"

namespace nnlib
{

/// \brief Log softmax for classification problems.
///
/// Best combined with NLL critic. Works only on 2D tensors.
template <typename T = NN_REAL_T>
class LogSoftMax : public Module<T>
{
public:
    using Module<T>::Module;

    LogSoftMax();

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;

protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;
};

}

NNRegisterType(LogSoftMax, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::LogSoftMax<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/logsoftmax.tpp"
#endif

#endif
