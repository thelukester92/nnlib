#ifndef TEST_RANDOM_H
#define TEST_RANDOM_H

#include "nnlib/util/random.h"
#include "nnlib/tensor.h"
using namespace nnlib;

void TestRandom()
{
	RandomEngine::seed();
	double r;
	
	r = Random<>::uniform(3.14);
	NNAssertGreaterThanOrEquals(r, 0, "Random::uniform(T) produced a value too small!");
	NNAssertLessThan(r, 3.14, "Random::uniform(T) produced a value too big!");
	
	r = Random<>::uniform(2.19, 3.14);
	NNAssertGreaterThanOrEquals(r, 2.19, "Random::uniform(T, T) produced a value too small!");
	NNAssertLessThan(r, 3.14, "Random::uniform(T, T) produced a value too big!");
	
	Tensor<> t(1000);
	for(auto &v : t)
		v = Random<>::normal();
	NNAssertAlmostEquals(t.mean(), 0.0, 1e-1, "Random::normal produced an unexpected mean!");
	NNAssertAlmostEquals(sqrt(t.variance()), 1.0, 1e-1, "Random::normal produced an unexpected standard deviation!");
	
	r = Random<>::normal(0, 1, 1);
	NNAssertGreaterThanOrEquals(r, -1.0, "Random::normal(T, T, T) produced a value too small!");
	NNAssertLessThanOrEquals(r, 1.0, "Random::uniform(T, T, T) produced a value too big!");
}

#endif
