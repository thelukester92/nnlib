#include "random.h"
#include "tensor.h"
using namespace nnlib;

/*
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
	std::normal_distribution<double> dist(mean, stddev);
	do
	{
		n = dist(m_engine);
	}
	while(fabs(n - mean) > cap);
	return n;
}

size_t Random::uniformInt(size_t n)
{
	return std::uniform_int_distribution<size_t>(0, n)(m_engine);
}

RandomIterator::RandomIterator(size_t n) : m_random(), m_buffer(n)
{
	reset();
}

void RandomIterator::reset()
{
	size_t n = m_buffer.size();
	for(size_t i = 0; i < n; ++i)
		m_buffer[i] = i;
	for(size_t i = n - 1; i > 0; --i)
		std::swap(m_buffer[i], m_buffer[m_random.uniformInt(i)]);
}

size_t *RandomIterator::begin()
{
	return m_buffer.begin();
}

size_t *RandomIterator::end()
{
	return m_buffer.end();
}
*/
