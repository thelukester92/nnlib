#ifndef NN_MODULE_HPP
#define NN_MODULE_HPP

#include "../core/tensor.hpp"

namespace nnlib
{

class Serialized;

/// The abtract base class for all neural network modules.
template <typename T = NN_REAL_T>
class Module
{
public:
    Module();
    Module(const Module &) = delete;
    virtual ~Module();

    Module &operator=(const Module &) = delete;

    /// Construct a copy of this module. Must be registered with NNRegisterType.
    Module *copy() const;

    /// Set whether this module is in training mode. Useful for modules like batchnorm that behave differently at evaluation time.
    virtual void training(bool training = true);

    /// Reset the internal state of this module. Useful for recurrent modules that have additional inner state.
    virtual void forget();

    /// \brief Save the current module to a serialized node.
    ///
    /// The load method is omitted; instead, a constructor taking a Serialized& should be implemented
    /// in subclasses of Module.
    virtual void save(Serialized &) const = 0;

    /// Evaluate the module and return the new output.
    virtual Tensor<T> &forward(const Tensor<T> &input) = 0;

    /// Take the derivative of the module and return the gradient of the input.
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) = 0;

    virtual Storage<Tensor<T> *> paramsList();
    virtual Storage<Tensor<T> *> gradList();
    virtual Storage<Tensor<T> *> stateList();

    Tensor<T> &params();
    Tensor<T> &grad();
    Tensor<T> &state();

    virtual Tensor<T> &output();
    const Tensor<T> &output() const;
    virtual Tensor<T> &inGrad();
    const Tensor<T> &inGrad() const;

protected:
    Tensor<T> m_output;
    Tensor<T> m_inGrad;

    Tensor<T> m_params;
    Tensor<T> m_grad;
    Tensor<T> m_state;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Module<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/module.tpp"
#endif

#endif
