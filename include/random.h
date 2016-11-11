#ifndef RANDOM_H
#define RANDOM_H

#include <random>

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
	
private:
	std::default_random_engine m_engine;
};

}

#endif
