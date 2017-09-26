#ifndef TEST_LOGISTIC_H
#define TEST_LOGISTIC_H

#include "nnlib/nn/logistic.h"
using namespace nnlib;

void TestLogistic()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({ 0.21416501695, 0.73105857863, 0.95851288069 }).resize(1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ 0.33659672493, -0.58983579972, 0.03976593824 }).resize(1, 3);
	
	Logistic<> map(0.5);
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().copy().addM(out, -1).square().sum() < 1e-9, "Logistic::forward failed!");
	NNAssert(map.inGrad().copy().addM(ing, -1).square().sum() < 1e-9, "Logistic::backward failed!");
}

#endif
