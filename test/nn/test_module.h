#ifndef TEST_MODULE_H
#define TEST_MODULE_H

#include "nnlib/nn/module.h"
#include "nnlib/core/tensor.h"
using namespace std;

template <typename T>
void TestModule(T &module)
{
	T copy(module);
	
	auto &p1 = module.parameters();
	auto &p2 = copy.parameters();
	
	NNAssertEquals(p1.shape(), p2.shape(), "Module::Module(const Module &) failed! Wrong parameter shape!");
	for(auto x = p1.begin(), y = p2.begin(), end = p1.end(); x != end; ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Module::Module(const Module &) failed! Wrong data!");
	
	p1.zeros();
	p2.ones();
	
	for(auto x = p1.begin(), y = p2.begin(), end = p1.end(); x != end; ++x, ++y)
	{
		NNAssertAlmostEquals(*x, 0, 1e-12, "Module::Module(const Module &) failed! Not a deep copy!");
		NNAssertAlmostEquals(*y, 1, 1e-12, "Module::Module(const Module &) failed! Not a deep copy!");
	}
	
	module = copy;
	p1 = module.parameters();
	for(auto &x : p1)
		NNAssertAlmostEquals(x, 1, 1e-12, "Module::operator=(const Module &) failed!");
	
	p1.zeros();
	for(auto &y : p2)
		NNAssertAlmostEquals(y, 1, 1e-12, "Module::operator=(const Module &) failed! Not a deep copy!");
}

template <typename T>
void TestSerializationOfModule(T &module)
{
	Serialized node(module);
	T serialized = node.as<T>();
	
	auto &p1 = module.parameters();
	auto &p2 = serialized.parameters();
	
	for(auto i = p1.begin(), j = p2.begin(), k = p1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed! Mismatching parameters.");
	
	auto tensor = module.output().copy();
	tensor.resize(module.inputs()).rand();
	
	RandomEngine::seed(0);
	auto &o1 = module.forward(tensor);
	
	RandomEngine::seed(0);
	auto &o2 = serialized.forward(tensor);
	
	for(auto i = o1.begin(), j = o2.begin(), k = o1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed! Different outputs for the same input.");
	
	std::clog << " (serialization stub) ";
	/*
	{
		Serialized node(&module);
		Module<> *generic = node.as<Module<> *>();
		
		auto &p1 = module.parameters();
		auto &p2 = generic->parameters();
		
		for(auto i = p1.begin(), j = p2.begin(), k = p1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Generic serialization failed! Mismatching parameters.");
		
		auto tensor = module.output().copy();
		tensor.resize(module.inputs()).rand();
		
		RandomEngine::seed(0);
		auto &o1 = module.forward(tensor);
		
		RandomEngine::seed(0);
		auto &o2 = generic->forward(tensor);
		
		for(auto i = o1.begin(), j = o2.begin(), k = o1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Generic serialization failed! Different outputs for the same input.");
		
		delete generic;
	}
	*/
}

#endif
