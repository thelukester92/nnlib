#ifndef UTIL_RANDOM_HPP
#define UTIL_RANDOM_HPP

#include <random>
#include "../core/error.hpp"

namespace nnlib
{

class RandomEngine
{
public:
    static RandomEngine &sharedEngine()
    {
        static RandomEngine e;
        return e;
    }

    inline RandomEngine();
    explicit inline RandomEngine(size_t seed);

    inline void seed(size_t seed = 0);
    inline std::default_random_engine &engine();

private:
    std::default_random_engine m_engine;
};

template <typename T = NN_REAL_T>
class Random
{
public:
    static Random &sharedRandom();

    explicit Random(size_t s = 0);
    explicit Random(RandomEngine *engine);
    ~Random();

    /// Uniform distribution of integers in [0, to)
    template <typename U = T>
    typename std::enable_if<std::is_integral<U>::value, T>::type uniform(T to = 100);

    /// Uniform distribution of integers in [from, to)
    template <typename U = T>
    typename std::enable_if<std::is_integral<U>::value, T>::type uniform(T from, T to);

    /// Uniform distribution of floating point numbers in [0, to)
    template <typename U = T>
    typename std::enable_if<!std::is_integral<U>::value, T>::type uniform(T to = 1);

    /// Uniform distribution of floating point numbers in [from, to)
    template <typename U = T>
    typename std::enable_if<!std::is_integral<U>::value, T>::type uniform(T from, T to);

    /// Normal distribution of floating point numbers.
    template <typename U = T>
    typename std::enable_if<!std::is_integral<U>::value, T>::type normal(T mean = 0.0, T stddev = 1.0);

    /// Normal distribution of floating point numbers, constrained to be in [mean-cap, mean+cap]
    template <typename U = T>
    typename std::enable_if<!std::is_integral<U>::value, T>::type normal(T mean, T stddev, T cap);

    /// Bernoulli distribution (binary).
    template <typename U = double>
    T bernoulli(U p);

private:
    RandomEngine *m_engine;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Random<NN_REAL_T>;
#endif

#include "detail/random.tpp"

#endif
