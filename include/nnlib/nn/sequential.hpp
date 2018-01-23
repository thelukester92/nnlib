#ifndef NN_SEQUENTIAL_HPP
#define NN_SEQUENTIAL_HPP

#include "container.hpp"

namespace nnlib
{

/// A standard feed-forward neural network module.
template <typename T = NN_REAL_T>
class Sequential : public Container<T>
{
public:
    using Container<T>::Container;
    using Container<T>::add;
    using Container<T>::components;

    template <typename ... Ms>
    Sequential(Ms... components) :
        Container<T>({ 1 }, { 1 }, components...)
    {}

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;
    virtual Tensor<T> &output() override;
    virtual Tensor<T> &inGrad() override;

    virtual const Storage<size_t> &inputShape() const override;
    virtual const Storage<size_t> &outputShape() const override;

protected:
    using Container<T>::m_components;
};

}

NNRegisterType(Sequential, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Sequential<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/sequential.tpp"
#endif

#endif
