#ifndef RANDOM_H
#define RANDOM_H

#include <random>

namespace nnlib
{

template <typename T = double>
class Random
{
public:
	Random() = delete;
	
	static void seed(size_t seed = 0)
	{
		m_engine.seed(0);
	}
	
	static T uniform(T to)
	{
		return std::uniform_real_distribution<T>(0, to)(m_engine);
	}
	
	static T uniform(T from, T to)
	{
		return std::uniform_real_distribution<T>(from, to)(m_engine);
	}
	
	static T normal(T mean = 0.0, T stddev = 1.0)
	{
		return std::normal_distribution<T>(mean, stddev)(m_engine);
	}
	
	static T normal(T mean, T stddev, T cap)
	{
		double n;
		std::normal_distribution<T> dist(mean, stddev);
		do
		{
			n = dist(m_engine);
		}
		while(fabs(n - mean) > cap);
		return n;
	}
private:
	static std::default_random_engine m_engine;
};

template <typename T>
std::default_random_engine Random<T>::m_engine = std::default_random_engine(std::random_device()());

template <>
class Random<size_t>
{
using T = size_t;
public:
	Random() = delete;
	
	static void seed(size_t seed = 0)
	{
		m_engine.seed(0);
	}
	
	static T uniform(T to)
	{
		return std::uniform_int_distribution<T>(0, to)(m_engine);
	}
	
	static T uniform(T from, T to)
	{
		return std::uniform_int_distribution<T>(from, to)(m_engine);
	}
private:
	static std::default_random_engine m_engine;
};

std::default_random_engine Random<size_t>::m_engine = std::default_random_engine(std::random_device()());

}

#endif
