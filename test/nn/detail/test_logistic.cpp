#include "../test_logistic.hpp"
#include "../test_map.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/logistic.hpp"
using namespace nnlib;

void TestLogistic()
{
	// Input, arbitrary
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -1.3, 1.0, 3.14 }).resize(1, 3);

	// Output gradient, arbitrary
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 2, -3, 1 }).resize(1, 3);

	// Output, fixed given input
	Tensor<NN_REAL_T> out = Tensor<NN_REAL_T>({ 0.21416501695, 0.73105857863, 0.95851288069 }).resize(1, 3);

	// Input gradient, fixed given input and output gradient
	Tensor<NN_REAL_T> ing = Tensor<NN_REAL_T>({ 0.33659672493, -0.58983579972, 0.03976593824 }).resize(1, 3);

	Logistic<NN_REAL_T> map;
	map.forward(inp);
	map.backward(inp, grd);

	NNAssert(math::sum(math::square(map.output() - out)) < 1e-9, "Logistic::forward failed!");
	NNAssert(math::sum(math::square(map.inGrad() - ing)) < 1e-9, "Logistic::backward failed!");

	TestMap("Logistic", map, inp);
}
