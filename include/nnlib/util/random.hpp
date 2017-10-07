#ifndef UTIL_RANDOM_HPP
#define UTIL_RANDOM_HPP

#include <random>
#include "../core/error.hpp"

namespace nnlib
{

class RandomEngine
{
public:
	RandomEngine() = delete;
	
	static void seed(size_t s = 0)
	{
		engine().seed(s);
	}
	
	static std::default_random_engine &engine()
	{
		static std::default_random_engine e = std::default_random_engine(std::random_device()());
		return e;
	}
};

template <typename T = double>
class Random
{
public:
	Random() = delete;
	
	/// Uniform distribution of integers in [0, to)
	template <typename U = T>
	static typename std::enable_if<std::is_integral<U>::value, T>::type uniform(T to = 100)
	{
		NNAssertGreaterThan(to, 0, "Expected positive value!");
		return std::uniform_int_distribution<T>(0, to - 1)(RandomEngine::engine());
	}
	
	/// Uniform distribution of integers in [from, to)
	template <typename U = T>
	static typename std::enable_if<std::is_integral<U>::value, T>::type uniform(T from, T to)
	{
		NNAssertGreaterThan(to, from, "Expected valid range!");
		return std::uniform_int_distribution<T>(from, to)(RandomEngine::engine());
	}
	
	/// Uniform distribution of floating point numbers in [0, to)
	template <typename U = T>
	static typename std::enable_if<!std::is_integral<U>::value, T>::type uniform(T to = 1)
	{
		NNAssertGreaterThan(to, 0, "Expected positive value!");
		return std::uniform_real_distribution<T>(0, to)(RandomEngine::engine());
	}
	
	/// Uniform distribution of floating point numbers in [from, to)
	template <typename U = T>
	static typename std::enable_if<!std::is_integral<U>::value, T>::type uniform(T from, T to)
	{
		NNAssertGreaterThan(to, from, "Expected valid range!");
		return std::uniform_real_distribution<T>(from, to)(RandomEngine::engine());
	}
	
	/// Normal distribution of floating point numbers.
	template <typename U = T>
	static typename std::enable_if<!std::is_integral<U>::value, T>::type normal(T mean = 0.0, T stddev = 1.0)
	{
		NNAssertGreaterThan(stddev, 0, "Expected positive standard deviation!");
		return std::normal_distribution<T>(mean, stddev)(RandomEngine::engine());
	}
	
	/// Normal distribution of floating point numbers, constrained to be in [mean-cap, mean+cap]
	template <typename U = T>
	static typename std::enable_if<!std::is_integral<U>::value, T>::type normal(T mean, T stddev, T cap)
	{
		NNAssertGreaterThan(stddev, 0, "Expected positive standard deviation!");
		T n;
		std::normal_distribution<T> dist(mean, stddev);
		do
		{
			n = dist(RandomEngine::engine());
		}
		while(fabs(n - mean) > cap);
		return n;
	}
	
	/// Bernoulli distribution (binary).
	template <typename U = double>
	static T bernoulli(U p)
	{
		return Random<U>::uniform() < p ? 1 : 0;
	}
};

}

#endif
