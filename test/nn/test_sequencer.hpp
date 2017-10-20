#ifndef TEST_SEQUENCER_H
#define TEST_SEQUENCER_H

#include "nnlib/nn/sequencer.hpp"
#include "nnlib/nn/lstm.hpp"
#include "test_module.hpp"
using namespace nnlib;

void TestSequencer()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ 8, 6, 0 }).resize(3, 1, 1);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 1, 0, -1 }).resize(3, 1, 1);
	
	// LSTM layer with specific weights and bias, arbitrary
	LSTM<> *lstm = new LSTM<>(1, 1);
	lstm->params().copy({
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
	
	// Reversed output, fixed given input and weights
	Tensor<> rOut = Tensor<>({ 0.3523252224669841, 0.1900534836059371, 0 }).resize(3, 1, 1);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ -0.01712796895, 0.00743178473, -0.30729831287 }).resize(3, 1, 1);
	
	// Reversed input gradient, fixed given input and output gradient
	Tensor<> rIng = Tensor<>({ -0.01058507197696686, -0.02743768942521932, 0.1646924317207443 }).resize(3, 1, 1);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		 0.73850659626,  0.00585685005,  0.00801518653,  0.11462669318,
		-0.00117516696, -0.01646944995, -0.02114722491, -0.05115589662,
		-0.00000416626, -0.08323296026, -0.25800409062,
		 0.18717939172, -0.00419251188,  0.00611825134,  0.00767284875
	});
	
	// Test forward and backward using the parameters and targets above
	
	Sequencer<> module(lstm);
	NNAssertEquals(lstm, &module.module(), "Sequencer::Sequencer failed!");
	
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssertLessThan(module.output().add(out, -1).square().sum(), 1e-9, "Sequencer::forward failed!");
	NNAssertLessThan(module.inGrad().add(ing, -1).square().sum(), 1e-9, "Sequencer::backward failed; wrong inGrad!");
	NNAssertLessThan(module.grad().addV(prg, -1).square().sum(), 1e-9, "Sequencer::backward failed; wrong grad!");
	
	module.forget();
	module.reverse(true);
	module.forward(inp);
	module.backward(inp, grd);
	
	std::cout << "got:\n" << module.output().select(1, 0) << std::endl << module.inGrad().select(1, 0) << std::endl;
	
	NNAssertLessThan(module.output().add(rOut, -1).square().sum(), 1e-9, "Sequencer::forward (reversed) failed!");
	NNAssertLessThan(module.inGrad().add(rIng, -1).square().sum(), 1e-9, "Sequencer::backward (reversed) failed; wrong inGrad!");
	
	{
		BatchNorm<> *b = new BatchNorm<>(10);
		Sequencer<> s(b);
		s.training(false);
		NNAssert(!b->isTraining(), "Sequencer::training failed!");
	}
	
	TestModule("Sequencer", module, inp);
}

#endif
