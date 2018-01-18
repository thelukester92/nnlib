#ifndef NN_SIN_TPP
#define NN_SIN_TPP

#include "../sin.hpp"
#include <math.h>

namespace nnlib
{

template <typename T>
T Sin<T>::forwardOne(const T &x)
{
    return sin(x);
}

template <typename T>
T Sin<T>::backwardOne(const T &x, const T &y)
{
    return cos(x);
}

}

#endif
