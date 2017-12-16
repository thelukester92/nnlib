#include "../test_logsoftmax.hpp"
#include "../test_module.hpp"
#include "nnlib/nn/logsoftmax.hpp"
using namespace nnlib;

void TestLogSoftMax()
{
	// Input, arbitrary
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 2, -4, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<NN_REAL_T> out = Tensor<NN_REAL_T>({ -4.56173148054, -2.26173148054, -0.12173148053 }).resize(1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<NN_REAL_T> ing = Tensor<NN_REAL_T>({ 2.01044395977, -3.89583003975, 1.88538607998 }).resize(1, 3);
	
	// Begin test
	
	LogSoftMax<NN_REAL_T> map;
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().addM(out, -1).square().sum() < 1e-9, "LogSoftMax::forward failed!");
	NNAssert(map.inGrad().addM(ing, -1).square().sum() < 1e-9, "LogSoftMax::backward failed!");
	
	TestModule("LogSoftMax", map, inp);
}
