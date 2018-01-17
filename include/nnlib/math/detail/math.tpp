#ifndef MATH_MATH_TPP
#define MATH_MATH_TPP

#include "../algebra.hpp"
#include "../math.hpp"
#include "nnlib/math/random.hpp"
#include <math.h>

namespace nnlib
{

namespace math
{

template <typename T>
T min(const Tensor<T> &x)
{
    T value = *x.begin();
    forEach([&](T x)
    {
        if(x < value)
            value = x;
    }, x);
    return value;
}

template <typename T>
T max(const Tensor<T> &x)
{
    T value = *x.begin();
    forEach([&](T x)
    {
        if(x > value)
            value = x;
    }, x);
    return value;
}

template <typename T>
T sum(const Tensor<T> &x)
{
    T value = 0;
    forEach([&](T x)
    {
        value += x;
    }, x);
    return value;
}

template <typename T>
T mean(const Tensor<T> &x)
{
    return sum(x) / x.size();
}

template <typename T>
T variance(const Tensor<T> &x, bool sample)
{
    T avg = mean(x), sum = 0;
    forEach([&](T x)
    {
        T diff = x - avg;
        sum += diff * diff;
    }, x);
    return sum / (x.size() - (sample ? 1 : 0));
}

template <typename T>
Tensor<T> &normalize(Tensor<T> &x, T from, T to)
{
    NNAssertLessThanOrEquals(from, to, "Invalid normalization range!");
    T small = min(x), large = max(x);
    return x.add(-small).scale((to - from) / (large - small)).add(from);
}

template <typename T>
Tensor<T> &&normalize(Tensor<T> &&x, T from, T to)
{
    return std::move(normalize(x, from, to));
}

template <typename T>
Tensor<T> &sum(const Tensor<T> &x, Tensor<T> &y, size_t dim)
{
    NNAssertLessThan(dim, x.dims(), "Invalid dimension for summation!");
    NNAssertGreaterThan(x.dims(), 1, "Cannot specify a summing dimension for a vector!");
    y.copy(x.select(dim, 0));
    for(size_t i = 1, n = x.size(dim); i < n; ++i)
    {
        forEach([&](T x, T &y)
        {
            y += x;
        }, x.select(dim, i), y);
    }
    return y;
}

template <typename T>
Tensor<T> &&sum(const Tensor<T> &x, Tensor<T> &&y, size_t dim)
{
    return std::move(sum(x, y, dim));
}

template <typename T>
Tensor<T> &clip(Tensor<T> &x, T min, T max)
{
    NNAssertLessThanOrEquals(min, max, "Invalid clipping range!");
    forEach([&](T &x)
    {
        if(x < min)
            x = min;
        else if(x > max)
            x = max;
    }, x);
    return x;
}

template <typename T>
Tensor<T> &&clip(Tensor<T> &&x, T min, T max)
{
    return std::move(clip(x, min, max));
}

template <typename T>
Tensor<T> &square(Tensor<T> &x)
{
    return pointwiseProduct(x, x);
}

template <typename T>
Tensor<T> &&square(Tensor<T> &&x)
{
    return std::move(pointwiseProduct(x, x));
}

template <typename T>
Tensor<T> &rand(Tensor<T> &x, T min, T max)
{
    forEach([&](T &v)
    {
        v = Random<T>::sharedRandom().uniform(min, max);
    }, x);
    return x;
}

template <typename T>
Tensor<T> &&rand(Tensor<T> &&x, T min, T max)
{
    return std::move(rand(x, min, max));
}

template <typename T>
Tensor<T> &randn(Tensor<T> &x, T mean, T stddev)
{
    forEach([&](T &v)
    {
        v = Random<T>::sharedRandom().normal(mean, stddev);
    }, x);
    return x;
}

template <typename T>
Tensor<T> &&randn(Tensor<T> &&x, T mean, T stddev)
{
    return std::move(randn(x, mean, stddev));
}

template <typename T>
Tensor<T> &randn(Tensor<T> &x, T mean, T stddev, T cap)
{
    forEach([&](T &v)
    {
        v = Random<T>::sharedRandom().normal(mean, stddev, cap);
    }, x);
    return x;
}

template <typename T>
Tensor<T> &&randn(Tensor<T> &&x, T mean, T stddev, T cap)
{
    return std::move(randn(x, mean, stddev, cap));
}

template <typename T>
Tensor<T> &bernoulli(Tensor<T> &x, T p)
{
    forEach([&](T &v)
    {
        v = Random<T>::sharedRandom().bernoulli(p);
    }, x);
    return x;
}

template <typename T>
Tensor<T> &&bernoulli(Tensor<T> &&x, T p)
{
    return std::move(bernoulli(x, p));
}

template <typename T>
Tensor<T> &pointwiseProduct(const Tensor<T> &x, Tensor<T> &y)
{
    NNAssertEquals(x.shape(), y.shape(), "Incompatible operands!");
    forEach([&](T x, T &y)
    {
        y *= x;
    }, x, y);
    return y;
}

template <typename T>
Tensor<T> &&pointwiseProduct(const Tensor<T> &x, Tensor<T> &&y)
{
    return std::move(pointwiseProduct(x, y));
}

template <typename T>
Tensor<T> &pointwiseProduct(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &z)
{
    NNAssertEquals(x.shape(), y.shape(), "Incompatible operands!");
    NNAssertEquals(x.shape(), z.shape(), "Incompatible operands!");
    forEach([&](T x, T y, T &z)
    {
        z = x * y;
    }, x, y, z);
    return z;
}

template <typename T>
Tensor<T> &&pointwiseProduct(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &&z)
{
    return std::move(pointwiseProduct(x, y, z));
}

}

}

#endif
