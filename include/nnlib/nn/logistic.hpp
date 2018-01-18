#ifndef NN_LOGISTIC_HPP
#define NN_LOGISTIC_HPP

#include "map.hpp"

namespace nnlib
{

/// Sigmoidal logistic activation function.
template <typename T = NN_REAL_T>
class Logistic : public Map<T>
{
public:
    using Map<T>::Map;

    virtual T forwardOne(const T &x) override;
    virtual T backwardOne(const T &x, const T &y) override;
};

}

NNRegisterType(Logistic, Module);

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Logistic<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/logistic.tpp"
#endif

#endif
