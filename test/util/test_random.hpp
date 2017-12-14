#ifndef TEST_RANDOM_H
#define TEST_RANDOM_H

#include "nnlib/util/random.hpp"
#include "nnlib/core/tensor.hpp"
using namespace nnlib;

void TestRandom()
{
	RandomEngine::seed();
	double r;
	
	r = Random<NN_REAL_T>::uniform(3.14);
	NNAssertGreaterThanOrEquals(r, 0, "Random::uniform(T) produced a value too small!");
	NNAssertLessThan(r, 3.14, "Random::uniform(T) produced a value too big!");
	
	r = Random<NN_REAL_T>::uniform(2.19, 3.14);
	NNAssertGreaterThanOrEquals(r, 2.19, "Random::uniform(T, T) produced a value too small!");
	NNAssertLessThan(r, 3.14, "Random::uniform(T, T) produced a value too big!");
	
	Tensor<NN_REAL_T> t(1000);
	for(auto &v : t)
		v = Random<NN_REAL_T>::normal();
	NNAssertAlmostEquals(t.mean(), 0.0, 1e-1, "Random::normal produced an unexpected mean!");
	NNAssertAlmostEquals(sqrt(t.variance()), 1.0, 1e-1, "Random::normal produced an unexpected standard deviation!");
	
	r = Random<NN_REAL_T>::normal(0, 1, 1);
	NNAssertGreaterThanOrEquals(r, -1.0, "Random::normal(T, T, T) produced a value too small!");
	NNAssertLessThanOrEquals(r, 1.0, "Random::uniform(T, T, T) produced a value too big!");
}

#endif
