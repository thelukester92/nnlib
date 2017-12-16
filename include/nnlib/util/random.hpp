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
	
	RandomEngine() :
		m_engine(std::random_device()())
	{}
	
	explicit RandomEngine(size_t seed) :
		m_engine(seed)
	{}
	
	void seed(size_t seed = 0)
	{
		m_engine.seed(seed);
	}
	
	std::default_random_engine &engine()
	{
		return m_engine;
	}
	
private:
	std::default_random_engine m_engine;
};

template <typename T = NN_REAL_T>
class Random
{
public:
	static Random sharedRandom()
	{
		static Random r(&RandomEngine::sharedEngine());
		return r;
	}
	
	explicit Random(size_t s = 0) :
		m_engine(new RandomEngine(s))
	{}
	
	explicit Random(RandomEngine *engine) :
		m_engine(engine)
	{}
	
	~Random()
	{
		if(m_engine != &RandomEngine::sharedEngine())
			delete m_engine;
	}
	
	/// Uniform distribution of integers in [0, to)
	template <typename U = T>
	typename std::enable_if<std::is_integral<U>::value, T>::type uniform(T to = 100)
	{
		NNAssertGreaterThan(to, 0, "Expected positive value!");
		return std::uniform_int_distribution<T>(0, to - 1)(m_engine->engine());
	}
	
	/// Uniform distribution of integers in [from, to)
	template <typename U = T>
	typename std::enable_if<std::is_integral<U>::value, T>::type uniform(T from, T to)
	{
		NNAssertGreaterThan(to, from, "Expected valid range!");
		return std::uniform_int_distribution<T>(from, to)(m_engine->engine());
	}
	
	/// Uniform distribution of floating point numbers in [0, to)
	template <typename U = T>
	typename std::enable_if<!std::is_integral<U>::value, T>::type uniform(T to = 1)
	{
		NNAssertGreaterThan(to, 0, "Expected positive value!");
		return std::uniform_real_distribution<T>(0, to)(m_engine->engine());
	}
	
	/// Uniform distribution of floating point numbers in [from, to)
	template <typename U = T>
	typename std::enable_if<!std::is_integral<U>::value, T>::type uniform(T from, T to)
	{
		NNAssertGreaterThan(to, from, "Expected valid range!");
		return std::uniform_real_distribution<T>(from, to)(m_engine->engine());
	}
	
	/// Normal distribution of floating point numbers.
	template <typename U = T>
	typename std::enable_if<!std::is_integral<U>::value, T>::type normal(T mean = 0.0, T stddev = 1.0)
	{
		NNAssertGreaterThan(stddev, 0, "Expected positive standard deviation!");
		return std::normal_distribution<T>(mean, stddev)(m_engine->engine());
	}
	
	/// Normal distribution of floating point numbers, constrained to be in [mean-cap, mean+cap]
	template <typename U = T>
	typename std::enable_if<!std::is_integral<U>::value, T>::type normal(T mean, T stddev, T cap)
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
	
	/// Bernoulli distribution (binary).
	template <typename U = double>
	T bernoulli(U p)
	{
		return Random<U>::sharedRandom().uniform() < p ? 1 : 0;
	}
	
private:
	RandomEngine *m_engine;
};

}

#endif
