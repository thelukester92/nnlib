#ifndef TEST_IDENTITY_H
#define TEST_IDENTITY_H

#include "nnlib/nn/identity.h"
#include "test_map.h"
using namespace nnlib;

void TestIdentity()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	TestMap<Identity<>>("Identity", inp, grd, inp, grd);
}

#endif
