#ifndef TRAITS_H
#define TRAITS_H

#include <type_traits>

namespace nnlib
{

template <bool C, typename T = void>
using EnableIf = typename std::enable_if<C, T>::type;

}

#endif
