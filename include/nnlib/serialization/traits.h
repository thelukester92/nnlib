#ifndef TRAITS_H
#define TRAITS_H

#include <type_traits>

namespace nnlib
{

template <bool C, typename T = void>
using EnableIf = typename std::enable_if<C, T>::type;

template <typename T, typename = int>
struct HasSerialize : std::false_type
{};

template <typename T>
struct HasSerialize<T, decltype(&T::serialize, 0)> : std::true_type
{};

template <typename T, typename = int>
struct HasLoadAndSave : std::false_type
{};

template <typename T>
struct HasLoadAndSave<T, decltype(&T::load, &T::save, 0)> : std::true_type
{};

}

#endif
