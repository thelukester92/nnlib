#ifndef NN_SIN_HPP
#define NN_SIN_HPP

#include "map.hpp"

namespace nnlib
{

/// Sinusoid activation function.
template <typename T = NN_REAL_T>
class Sin : public Map<T>
{
public:
    using Map<T>::Map;

    virtual T forwardOne(const T &x) override;
    virtual T backwardOne(const T &x, const T &y) override;
};

}

NNRegisterType(Sin, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Sin<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/sin.tpp"
#endif

#endif
