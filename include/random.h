#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include "tensor.h"

namespace nnlib
{

class Random
{
public:
	Random();
	Random(size_t seed);
	
	double uniform(double a = 0.0, double b = 1.0);
	double normal(double mean = 0.0, double stddev = 1.0);
	double normal(double mean, double stddev, double cap);
	size_t uniformInt(size_t n = std::numeric_limits<size_t>::max());
	
	/// Set all elements in a tensor to random values drawn from a normal distribution.
	template <typename T>
	void fillNormal(Tensor<T> &t, double mean = 0.0, double stddev = 1.0, double cap = 3.0)
	{
		for(auto &i : t)
			i = normal(mean, stddev, cap);
	}
	
private:
	std::default_random_engine m_engine;
};

class RandomIterator
{
public:
	RandomIterator(size_t n);
	void reset();
	size_t *begin();
	size_t *end();
private:
	Random m_random;
	Vector<size_t> m_buffer;
};

}

#endif
