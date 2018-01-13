#ifndef CORE_TENSOR_UTIL_HPP
#define CORE_TENSOR_UTIL_HPP

#include "../tensor.hpp"

namespace nnlib
{

/// A more efficient way apply a function to each element in one or more tensors.
template <typename F, typename T, typename ... Ts>
void forEach(F func, T &first, Ts &...ts);

}

#include "tensor_util.tpp"

#endif
