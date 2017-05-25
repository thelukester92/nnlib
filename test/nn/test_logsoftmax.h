#ifndef TEST_LOGSOFTMAX_H
#define TEST_LOGSOFTMAX_H

#include "nnlib/nn/logsoftmax.h"
using namespace nnlib;

void TestLogSoftMax()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({ -4.56173148054, -2.26173148054, -0.12173148053 }).resize(1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ 1.97911208047, -2.68749011924, 0.11461392001 }).resize(1, 3);
	
	// Begin test
	
	LogSoftMax<> map(3, 1);
	map.forward(inp);
	map.backward(inp, grd);
	
	NNHardAssert(map.output().addM(out, -1).square().sum() < 1e-9, "LogSoftMax::forward failed!");
	NNHardAssert(map.inGrad().addM(ing, -1).square().sum() < 1e-9, "LogSoftMax::backward failed!");
	
	map.inputs({ 3, 4 });
	NNHardAssert(map.inputs() == map.outputs(), "LogSoftMax::inputs failed to resize outputs!");
	
	map.outputs({ 12, 3 });
	NNHardAssert(map.inputs() == map.outputs(), "LogSoftMax::outputs failed to resize inputs!");
	
	bool ok = true;
	try
	{
		map.resize({ 3, 4 }, { 4, 3 });
		ok = false;
	}
	catch(const std::runtime_error &e) {}
	NNHardAssert(ok, "LogSoftMax::resize allowed unequal inputs and outputs!");
	
	LogSoftMax<> *deserialized = nullptr;
	Archive::fromString((Archive::toString() << map).str()) >> deserialized;
	NNHardAssert(
		deserialized != nullptr && map.inputs() == deserialized->inputs() && map.outputs() == deserialized->outputs(),
		"LogSoftMax::save and/or Identity::load failed!"
	);
	
	delete deserialized;
}

#endif
