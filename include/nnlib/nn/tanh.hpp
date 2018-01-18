#ifndef NN_TANH_HPP
#define NN_TANH_HPP

#include "map.hpp"

namespace nnlib
{

/// Hyperbolic tangent activation function.
template <typename T = NN_REAL_T>
class TanH : public Map<T>
{
public:
    using Map<T>::Map;

    virtual T forwardOne(const T &x) override;
    virtual T backwardOne(const T &x, const T &y) override;
};

}

NNRegisterType(TanH, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::TanH<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/tanh.tpp"
#endif

#endif
