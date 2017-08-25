#ifndef TEST_RECURRENT_H
#define TEST_RECURRENT_H

#include "nnlib/nn/recurrent.h"
#include "test_module.h"
using namespace nnlib;

void TestRecurrent()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({
		-5, 10,
		15, -20,
		3, 4
	}).resize(3, 1, 2);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({
		1, 2, 3,
		3, -4, 5,
		-3, 2, -7
	}).resize(3, 1, 3);
	
	// Recurrent layer with specific weights and bias, arbitrary
	Recurrent<> module(2, 3);
	module.parameters().copy({
		// inpMod: 9
		0.1, 0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, 0.9,
		
		// memMod: 12
		1, 0, 1, 2, 1, 0, 1, 2, 3, 0.1, -0.1, 0.5,
		
		// outMod: 12
		0.2, 0.1, -0.2, 0.1, 0.2, -0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.3
	});
	
	// Output, fixed given input and weights
	Tensor<> out = Tensor<>({
		0.68048, 0.83965, 0.96609,
		0.99999, 0.99986, -0.99694,
		1, 1, -0.98688
	}).resize(3, 1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({
		0.13982, 0.24855,
		0.00783, 0.02819,
		-0.00365, -0.02738
	}).resize(3, 1, 2);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		-0.43214, 0.23833, -1.99021, 1.16713, 0.58568, 3.31054, 0.11996, 0.33347, 0.21769, 0.72889, -0.10012,
		-0.54170, 0.34309, 0.59968, -0.51560, 0.20058, 0.63773, -0.41713, 0.11996, 0.33347, 0.21769, -2.73774,
		-3.02950, -8.48463, 2.52406, 2.76177, -3.58439, 3.16826, 3.47373, -6.52365, 0.53699, 0.58888, 0.04810
	});
	
	// Test forward and backward using the parameters and targets above
	
	Tensor<> &state = module.state();
	Tensor<> states(inp.size(0), state.size());
	Tensor<> outputs(3, 1, 3);
	Tensor<> inGrads(3, 1, 2);
	
	for(size_t i = 0; i < inp.size(0); ++i)
	{
		outputs.select(0, i).copy(module.forward(inp.select(0, i)));
		states.select(0, i).copy(state);
	}
	
	for(size_t i = inp.size(0); i > 0; --i)
	{
		state.copy(states.select(0, i - 1));
		inGrads.select(0, i - 1).copy(module.backward(inp.select(0, i - 1), grd.select(0, i - 1)));
	}
	
	NNAssert(outputs.add(out, -1).square().sum() < 1e-6, "Recurrent::forward failed!");
	NNAssert(inGrads.add(ing, -1).square().sum() < 1e-6, "Recurrent::backward failed; wrong inGrad!");
	NNAssert(module.grad().addV(prg, -1).square().sum() < 1e-6, "Recurrent::backward failed; wrong grad!");
	
	module.batch(32);
	NNAssert(module.batch() == 32, "Recurrent::batch failed!");
	
	bool ok = true;
	try
	{
		module.add(nullptr);
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "Recurrent::add failed to throw an error!");
	
	ok = true;
	try
	{
		module.remove(0);
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "Recurrent::remove failed to throw an error!");
	
	ok = true;
	try
	{
		module.clear();
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "Recurrent::clear failed to throw an error!");
	
	Storage<size_t> dims = { 3, 6 };
	
	module.inputs(dims);
	NNAssertEquals(module.inputs(), dims, "Recurrent::inputs failed!");
	
	module.outputs(dims);
	NNAssertEquals(module.outputs(), dims, "Recurrent::outputs failed!");
	
	/// \todo remove this hack; tensor flattening from state broke this
	module = Recurrent<>(module);
	
	TestSerializationOfModule(module);
	TestModule(module);
}

#endif
