#ifndef TEST_LSTM_H
#define TEST_LSTM_H

#include "nnlib/nn/lstm.hpp"
#include "test_module.hpp"
using namespace nnlib;

void TestLSTM()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ 8, 6, 0 }).resize(3, 1, 1);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 1, 0, -1 }).resize(3, 1, 1);
	
	// LSTM layer with specific weights and bias, arbitrary
	LSTM<> module(1, 1);
	module.params().copy({
		// inpGate (x, y, h, b)
		-0.2, 0.5, 0.1, 0.0,
		
		// fgtGate (x, y, h, b)
		0.75, -0.6, 0.25, 0.0,
		
		// inpMod (x, y, b)
		1.0, -0.7, 0.0,
		
		// outGate (x, y, h', b)
		0.3, 0.3, -0.75, 0.0
	});
	
	// Output, fixed given input and weights
	Tensor<> out = Tensor<>({ 0.15089258930, 0.32260369939, 0.03848645247 }).resize(3, 1, 1);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ -0.01712796895, 0.00743178473, -0.30729831287 }).resize(3, 1, 1);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		 0.73850659626,  0.00585685005,  0.00801518653,  0.11462669318,
		-0.00117516696, -0.01646944995, -0.02114722491, -0.05115589662,
		-0.00000416626, -0.08323296026, -0.25800409062,
		 0.18717939172, -0.00419251188,  0.00611825134,  0.00767284875
	});
	
	// Test forward and backward using the parameters and targets above
	
	Tensor<> states(inp.size(0), 0);
	
	Tensor<> outputs(3, 1, 1);
	Tensor<> inGrads(3, 1, 1);
	
	for(size_t i = 0; i < inp.size(0); ++i)
	{
		outputs.select(0, i).copy(module.forward(inp.select(0, i)));
		if(i == 0)
			states.resizeDim(1, module.state().size());
		states.select(0, i).copy(module.state());
	}
	
	for(size_t i = inp.size(0); i > 0; --i)
	{
		module.state().copy(states.select(0, i - 1));
		inGrads.select(0, i - 1).copy(module.backward(inp.select(0, i - 1), grd.select(0, i - 1)));
	}
	
	NNAssertLessThan(outputs.add(out, -1).square().sum(), 1e-6, "LSTM::forward failed!");
	NNAssertLessThan(inGrads.add(ing, -1).square().sum(), 1e-6, "LSTM::backward failed; wrong inGrad!");
	NNAssertLessThan(module.grad().addV(prg, -1).square().sum(), 1e-6, "LSTM::backward failed; wrong grad!");
	
	module.gradClip(0.03);
	module.forget();
	
	for(size_t i = 0; i < inp.size(0); ++i)
	{
		outputs.select(0, i).copy(module.forward(inp.select(0, i)));
		states.select(0, i).copy(module.state());
	}
	
	for(size_t i = inp.size(0); i > 0; --i)
	{
		module.state().copy(states.select(0, i - 1));
		inGrads.select(0, i - 1).copy(module.backward(inp.select(0, i - 1), grd.select(0, i - 1)));
	}
	
	NNAssert(inGrads.add(ing.clip(-0.03, 0.03), -1).square().sum() < 1e-6, "LSTM::gradClip failed!");
	
	TestModule("LSTM", module, inp.select(0, 0));
}

#endif
