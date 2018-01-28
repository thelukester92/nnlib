#ifndef NN_PRELU_HPP
#define NN_PRELU_HPP

#include "map.hpp"

namespace nnlib
{

/// (Dynamically) parameterized rectified linear activation function.
template <typename T = NN_REAL_T>
class PReLU : public Module<T>
{
public:
    PReLU(size_t size);
    PReLU(const PReLU &module);
    PReLU(const Serialized &node);

    PReLU &operator=(const PReLU &module);

    virtual void save(Serialized &node) const override;

    T leak(size_t i) const;

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;

    virtual Storage<Tensor<T> *> paramsList() override;
    virtual Storage<Tensor<T> *> gradList() override;

protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;

private:
    Tensor<T> m_leaks;
    Tensor<T> m_grads;
};

}

NNRegisterType(PReLU, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::PReLU<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/PReLU.tpp"
#endif

#endif
