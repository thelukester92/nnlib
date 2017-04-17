#ifndef RANDOM_H
#define RANDOM_H

#include <random>

namespace nnlib
{

class RandomEngine
{
public:
	RandomEngine() = delete;
	
	static void seed(size_t s = 0)
	{
		m_engine.seed(s);
	}
	
	static std::default_random_engine &engine()
	{
		return m_engine;
	}
	
private:
	static std::default_random_engine m_engine;
};

std::default_random_engine RandomEngine::m_engine = std::default_random_engine(std::random_device()());

template <typename T = double>
class Random
{
public:
	Random() = delete;
	
	template <typename U = T>
	static typename std::enable_if<std::is_integral<U>::value, T>::type uniform(T to)
	{
		return std::uniform_int_distribution<T>(0, to)(RandomEngine::engine());
	}
	
	template <typename U = T>
	static typename std::enable_if<std::is_integral<U>::value, T>::type uniform(T from, T to)
	{
		return std::uniform_int_distribution<double>(from, to)(RandomEngine::engine());
	}
	
	template <typename U = T>
	static typename std::enable_if<!std::is_integral<U>::value, T>::type uniform(T to)
	{
		return std::uniform_real_distribution<T>(0, to)(RandomEngine::engine());
	}
	
	template <typename U = T>
	static typename std::enable_if<!std::is_integral<U>::value, T>::type uniform(T from, T to)
	{
		return std::uniform_real_distribution<double>(from, to)(RandomEngine::engine());
	}
	
	template <typename U = T>
	static typename std::enable_if<!std::is_integral<U>::value, T>::type normal(T mean = 0.0, T stddev = 1.0)
	{
		return std::normal_distribution<T>(mean, stddev)(RandomEngine::engine());
	}
	
	template <typename U = T>
	static typename std::enable_if<!std::is_integral<U>::value, T>::type normal(T mean, T stddev, T cap)
	{
		double n;
		std::normal_distribution<T> dist(mean, stddev);
		do
		{
			n = dist(RandomEngine::engine());
		}
		while(fabs(n - mean) > cap);
		return n;
	}
};

}

#endif
