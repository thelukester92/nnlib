#include "../test_container.hpp"
#include "../test_sequential.hpp"
#include "nnlib/nn/sequential.hpp"
#include "nnlib/nn/batchnorm.hpp"
#include "nnlib/nn/linear.hpp"
using namespace nnlib;

void TestSequential()
{
	// Input, arbitrary
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({
		-5, 10,
		15, -20
	}).resize(2, 2);

	// Output gradient, arbitrary
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({
		1, 2,
		-4, -3,
	}).resize(2, 2);

	// Linear layer with weights and bias, arbitrary
	Linear<NN_REAL_T> *linear = new Linear<NN_REAL_T>(2, 3);
	linear->weights().copy({
		-0.50667886785403, 0.32759806987449, 0.65755833165511,
		0.10750948608753, -0.4340286671044, 0.23516327870398
	});
	linear->bias().copy({ -0.025146033726567, 0.6293391970186, -0.60999610504332 });

	// Second linear layer with weights and bias, arbitrary
	Linear<NN_REAL_T> *linear2 = new Linear<NN_REAL_T>(3, 2);
	linear2->weights().copy({
		-0.26131507397613, -0.12238678004357,
		-0.25173198611324, -0.52631174306551,
		-0.15799479364335, 0.076954514512593
	});
	linear2->bias().copy({ 0.089687510972433, -0.1780101224473 });

	// Output, fixed given input
	Tensor<NN_REAL_T> out = Tensor<NN_REAL_T>({
		0.74408910465583, 2.0796612294443,
		-1.6553227544948, -6.1176610608337
	}).resize(2, 2);

	// Input gradient, fixed given input and output gradient
	Tensor<NN_REAL_T> ing = Tensor<NN_REAL_T>({
		-0.17356654756232, 0.510757516282,
		0.39523702099164, -0.87616246292012
	}).resize(2, 2);

	// Parameter gradient, fixed given the input and output gradient
	Tensor<NN_REAL_T> prg = Tensor<NN_REAL_T>({
		23.716752710845, 45.309724965964, 6.037163288625,
		-33.309299061337, -64.760818195432, -8.0631702668939,
		0.90633200197196, 1.2815077014052, 0.39702986641745,
		42.68541825957, 36.493242652701,
		-62.244472172293, -53.369526408467,
		-19.746608159094, -16.742649839669,
		-3, -1
	});

	// Test forward and backward using the parameters and targets above

	Sequential<NN_REAL_T> module(linear, linear2);
	module.forward(inp);
	module.backward(inp, grd);

	NNAssert((module.output() - out).square().sum() < 1e-9, "Sequential::forward failed!");
	NNAssert((module.inGrad() - ing).square().sum() < 1e-9, "Sequential::backward failed; wrong inGrad!");
	NNAssert((module.grad() - prg).square().sum() < 1e-9, "Sequential::backward failed; wrong grad!");

	NNAssertEquals(module.component(0), linear, "Sequential::component failed to get the correct component!");
	NNAssertEquals(module.components(), 2, "Sequential::components failed!");
	NNAssertEquals(module.remove(0), linear, "Sequential::remove failed to return the removed component!");

	module.clear();
	NNAssertEquals(module.components(), 0, "Sequential::clear failed!");

	module.add(linear);
	NNAssertEquals(module.paramsList(), linear->paramsList(), "Sequential::paramsList failed!");
	NNAssertEquals(module.gradList(), linear->gradList(), "Sequential::gradList failed!");

	{
		BatchNorm<NN_REAL_T> *b = new BatchNorm<NN_REAL_T>(10);
		Sequential<NN_REAL_T> s(b);
		s.training(false);
		NNAssert(!b->isTraining(), "Sequential::training failed!");
	}

	Sequential<NN_REAL_T> module2(new Linear<NN_REAL_T>(5, 10), new Linear<NN_REAL_T>(10, 2));
	TestContainer("Sequential", module2, Tensor<NN_REAL_T>(100, 5).rand());
}
