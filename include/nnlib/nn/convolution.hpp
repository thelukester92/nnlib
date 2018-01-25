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
    static Storage<size_t> calcInputShape(size_t kernelCount, size_t inChannels, const Storage<size_t> &kernelShape, const Storage<size_t> &stride, bool interleaved);
    static Storage<size_t> calcOutputShape(size_t kernelCount, size_t inChannels, const Storage<size_t> &stride, const Storage<size_t> &pad, bool interleaved);

    Convolution(size_t kernelCount, size_t inChannels, const Storage<size_t> &kernelShape, const Storage<size_t> &stride = { 1, 1 }, const Storage<size_t> &pad = { 0, 0 }, bool interleaved = false);
    Convolution(size_t kernelCount, size_t inChannels, size_t kernelShape, size_t stride = 1, size_t pad = 0, bool interleaved = false);
    Convolution(const Convolution &module);
    Convolution(const Serialized &node);

    Convolution &operator=(const Convolution &module);

    size_t kernelCount() const;
    size_t inChannels() const;
    Storage<size_t> kernelShape() const;
    const Storage<size_t> &stride() const;
    const Storage<size_t> &pad() const;
    bool interleaved() const;

    Tensor<T> kernels();
    Tensor<T> bias();

    Convolution &reset();

    virtual void save(Serialized &node) const override;

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;

    virtual Storage<Tensor<T> *> paramsList() override;
    virtual Storage<Tensor<T> *> gradList() override;

protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;

    Tensor<T> m_kernels;
    Tensor<T> m_kernelsGrad;
    Tensor<T> m_bias;
    Tensor<T> m_biasGrad;

    Storage<size_t> m_stride;
    Storage<size_t> m_pad;
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
