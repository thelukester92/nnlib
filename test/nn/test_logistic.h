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
	Tensor<> ing = Tensor<>({ 0.16829836246, 0.19661193324, 0.03976593824 }).resize(1, 3);
	
	// Begin test
	
	Logistic<> tanh(3, 1);
	tanh.forward(inp);
	tanh.backward(inp, grd);
	
	NNHardAssert(tanh.output().addM(out, -1).sum() < 1e-9, "Logistic::forward failed!");
	NNHardAssert(tanh.inGrad().addM(ing, -1).sum() < 1e-9, "Logistic::backward failed!");
}
