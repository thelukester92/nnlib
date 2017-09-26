#ifndef TEST_MAP_H
#define TEST_MAP_H

#include "test_module.h"
using namespace nnlib;

template <typename M>
void TestMap(const std::string &name, M &map, const Tensor<> &inp, const Tensor<> &grd, const Tensor<> &out, const Tensor<> &ing)
{
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().copy().addM(out, -1).square().sum() < 1e-9, name + "::forward failed!");
	NNAssert(map.inGrad().copy().addM(ing, -1).square().sum() < 1e-9, name + "::backward failed!");
	
	map.resizeInputs(3, 4);
	NNAssert(map.inputShape() == map.outputShape(), name + "::resizeInputs failed to resize outputs!");
	
	map.resizeOutputs(12, 3);
	NNAssert(map.inputShape() == map.outputShape(), name + "::resizeOutputs failed to resize inputs!");
	
	bool ok = true;
	try
	{
		map.resize({ 3, 4 }, { 4, 3 });
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, name + "::resize allowed unequal inputs and outputs!");
	
	map.resize({ 3, 4 }, { 3, 4 });
	NNAssertEquals(map.inputShape(), map.outputShape(), name + "::resize failed!");
	
	map.forget();
	NNAssertAlmostEquals(map.output().sum(), 0.0, 1e-12, "Identity::forget failed!");
	
	TestModule(map);
}

template <typename M>
void TestMap(const std::string &name, const Tensor<> &inp, const Tensor<> &grd, const Tensor<> &out, const Tensor<> &ing)
{
	M map;
	TestMap(name, map, inp, grd, out, ing);
}

#endif
