#include "nnlib/nn/identity.h"
using namespace nnlib;

void TestIdentity()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = inp;
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = grd;
	
	// Begin test
	
	Identity<> map(3, 1);
	map.forward(inp);
	map.backward(inp, grd);
	
	NNHardAssert(map.output().addM(out, -1).square().sum() < 1e-9, "Identity::forward failed!");
	NNHardAssert(map.inGrad().addM(ing, -1).square().sum() < 1e-9, "Identity::backward failed!");
	
	map.inputs({ 3, 4 });
	NNHardAssert(map.inputs() == map.outputs(), "Identity::inputs failed to resize outputs!");
	
	map.outputs({ 12, 3 });
	NNHardAssert(map.inputs() == map.outputs(), "Identity::outputs failed to resize inputs!");
	
	bool ok = true;
	try
	{
		map.resize({ 3, 4 }, { 4, 3 });
		ok = false;
	}
	catch(const std::runtime_error &e) {}
	NNHardAssert(ok, "Identity::resize allowed unequal inputs and outputs!");
	
	Identity<> *deserialized = nullptr;
	Archive::fromString((Archive::toString() << map).str()) >> deserialized;
	NNHardAssert(
		deserialized != nullptr && map.inputs() == deserialized->inputs() && map.outputs() == deserialized->outputs(),
		"Identity::save and/or Identity::load failed!"
	);
	
	delete deserialized;
}