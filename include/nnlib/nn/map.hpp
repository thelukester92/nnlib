#ifndef NN_MAP_HPP
#define NN_MAP_HPP

#include "module.hpp"

namespace nnlib
{

/// Abstract base class for pointwise functions on inputs, also known as activation functions.
template <typename T = NN_REAL_T>
class Map : public Module<T>
{
public:
    using Module<T>::Module;

    virtual T forwardOne(const T &x) = 0;
    virtual T backwardOne(const T &x, const T &y) = 0;

    virtual Tensor<T> &forward(const Tensor<T> &input) override;
    virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;

protected:
    using Module<T>::m_output;
    using Module<T>::m_inGrad;
};

}

NNRegisterType(Map, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Map<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/map.tpp"
#endif

#endif
