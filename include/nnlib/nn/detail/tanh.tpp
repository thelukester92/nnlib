#ifndef NN_TANH_TPP
#define NN_TANH_TPP

#include "../tanh.hpp"
#include <math.h>

namespace nnlib
{

template <typename T>
T TanH<T>::forwardOne(const T &x)
{
    return tanh(x);
}

template <typename T>
T TanH<T>::backwardOne(const T &x, const T &y)
{
    return 1.0 - y * y;
}

}

#endif
