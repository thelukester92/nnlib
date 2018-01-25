#ifndef NN_CONVOLUTION_HPP
#define NN_CONVOLUTION_HPP

#include "module.hpp"

namespace nnlib
{

/// A 2D convolutional layer.
template <typename T = NN_REAL_T>
class Convolution : public Module<T>
{
public:
    Convolution(size_t filters, size_t channels, size_t kWidth, size_t kHeight, size_t strideX = 1, size_t strideY = 1, bool pad = false, bool interleaved = false);
    Convolution(const Convolution &module);
    Convolution(const Serialized &node);

    Convolution &operator=(const Convolution &module);

    size_t filters() const;
    size_t channels() const;
    size_t kernelHeight() const;
    size_t kernelWidth() const;
    size_t strideY() const;
    size_t strideX() const;
    bool padded() const;
    bool interleaved() const;

    Convolution &reset();

    virtual void save(Serialized &node) const override;

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;

    virtual Storage<Tensor<T> *> paramsList() override;
    virtual Storage<Tensor<T> *> gradList() override;

protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;

    Tensor<T> m_filters;
    Tensor<T> m_filtersGrad;

    size_t m_strideX;
    size_t m_strideY;

    bool m_pad;
    bool m_interleaved;
};

}

NNRegisterType(Convolution, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Convolution<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/convolution.tpp"
#endif

#endif
