#ifndef NN_LINEAR_HPP
#define NN_LINEAR_HPP

#include "module.hpp"

namespace nnlib
{

template <typename T>
class Linear;

template <typename T>
void swap(Linear<T> &, Linear<T> &);

/// A standard feed-forward layer that returns a linear combination of inputs.
template <typename T = NN_REAL_T>
class Linear : public Module<T>
{
public:
    Linear(size_t inps, size_t outs, bool bias = true);
    Linear(const Linear &module);
    Linear(const Serialized &node);

    Linear &operator=(Linear module);

    friend void swap <> (Linear &a, Linear &b);

    bool biased() const;

    Linear &reset();

    size_t inputs() const;
    size_t outputs() const;

    Tensor<T> weights();
    Tensor<T> bias();

    virtual void save(Serialized &node) const override;

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;

    virtual Storage<Tensor<T> *> paramsList() override;
    virtual Storage<Tensor<T> *> gradList() override;

protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;

    Tensor<T> m_weights;
    Tensor<T> m_weightsGrad;

    bool m_useBias;
    Tensor<T> m_bias;
    Tensor<T> m_biasGrad;

    Tensor<T> m_ones;
};

}

NNRegisterType(Linear, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Linear<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/linear.tpp"
#endif

#endif
