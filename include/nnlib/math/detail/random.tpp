#ifndef MATH_RANDOM_TPP
#define MATH_RANDOM_TPP

#include "../random.hpp"

namespace nnlib
{

RandomEngine::RandomEngine() :
    m_engine(std::random_device()())
{}

RandomEngine::RandomEngine(size_t seed) :
    m_engine(seed)
{}

void RandomEngine::seed(size_t seed)
{
    m_engine.seed(seed);
}

std::default_random_engine &RandomEngine::engine()
{
    return m_engine;
}

template <typename T>
Random<T> &Random<T>::sharedRandom()
{
    static Random r(&RandomEngine::sharedEngine());
    return r;
}

template <typename T>
Random<T>::Random(size_t s) :
    m_engine(new RandomEngine(s))
{}

template <typename T>
Random<T>::Random(RandomEngine *engine) :
    m_engine(engine)
{}

template <typename T>
Random<T>::~Random()
{
    if(m_engine != &RandomEngine::sharedEngine())
        delete m_engine;
}

template <typename T>
template <typename U>
typename std::enable_if<std::is_integral<U>::value, T>::type Random<T>::uniform(T to)
{
    NNAssertGreaterThan(to, 0, "Expected positive value!");
    return std::uniform_int_distribution<T>(0, to - 1)(m_engine->engine());
}

template <typename T>
template <typename U>
typename std::enable_if<std::is_integral<U>::value, T>::type Random<T>::uniform(T from, T to)
{
    NNAssertGreaterThan(to, from, "Expected valid range!");
    return std::uniform_int_distribution<T>(from, to)(m_engine->engine());
}

template <typename T>
template <typename U>
typename std::enable_if<!std::is_integral<U>::value, T>::type Random<T>::uniform(T to)
{
    NNAssertGreaterThan(to, 0, "Expected positive value!");
    return std::uniform_real_distribution<T>(0, to)(m_engine->engine());
}

template <typename T>
template <typename U>
typename std::enable_if<!std::is_integral<U>::value, T>::type Random<T>::uniform(T from, T to)
{
    NNAssertGreaterThan(to, from, "Expected valid range!");
    return std::uniform_real_distribution<T>(from, to)(m_engine->engine());
}

template <typename T>
template <typename U>
typename std::enable_if<!std::is_integral<U>::value, T>::type Random<T>::normal(T mean, T stddev)
{
    NNAssertGreaterThan(stddev, 0, "Expected positive standard deviation!");
    return std::normal_distribution<T>(mean, stddev)(m_engine->engine());
}

template <typename T>
template <typename U>
typename std::enable_if<!std::is_integral<U>::value, T>::type Random<T>::normal(T mean, T stddev, T cap)
{
    NNAssertGreaterThan(stddev, 0, "Expected positive standard deviation!");
    T n;
    std::normal_distribution<T> dist(mean, stddev);
    do
    {
        n = dist(m_engine->engine());
    }
    while(fabs(n - mean) > cap);
    return n;
}

template <typename T>
template <typename U>
T Random<T>::bernoulli(U p)
{
    return Random<U>::sharedRandom().uniform() < p ? 1 : 0;
}

}

#endif
