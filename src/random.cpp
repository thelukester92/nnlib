#include "random.h"
using namespace nnlib;

Random::Random() : m_engine(std::random_device()())
{}

Random::Random(size_t seed) : m_engine(seed)
{}

double Random::uniform(double a, double b)
{
	return std::uniform_real_distribution<double>(a, b)(m_engine);
}

double Random::normal(double mean, double stddev)
{
	return std::normal_distribution<double>(mean, stddev)(m_engine);
}

double Random::normal(double mean, double stddev, double cap)
{
	double n;
	do
	{
		n = std::normal_distribution<double>(mean, stddev)(m_engine);
	}
	while(fabs(n - mean) > cap);
	return n;
}

size_t Random::uniformInt(size_t n)
{
	return std::uniform_int_distribution<size_t>(0, n)(m_engine);
}
