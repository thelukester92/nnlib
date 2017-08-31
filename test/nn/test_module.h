#ifndef TEST_MODULE_H
#define TEST_MODULE_H

#include "nnlib/nn/container.h"
#include "nnlib/nn/module.h"
using namespace std;
using namespace nnlib;

template <typename T>
bool TestModule_equalParams(Module<T> &m1, Module<T> &m2)
{
	auto &p1 = m1.parameters();
	auto &p2 = m2.parameters();
	
	if(p1.shape() != p2.shape())
		return false;
	
	for(auto x = p1.begin(), y = p2.begin(), end = p1.end(); x != end; ++x, ++y)
	{
		if(std::abs(*x - *y) > 1e-12)
			return false;
	}
	
	return true;
}

template <typename T>
bool TestModule_notShared(Module<T> &m1, Module<T> &m2)
{
	return !m1.parameters().sharedWith(m2.parameters());
}

template <typename T>
bool TestModule_equalOutput(Module<T> &m1, Module<T> &m2)
{
	if(m1.inputs() != m2.inputs() || m1.outputs() != m2.outputs())
		return false;
	
	Tensor<T> input = Tensor<T>(m1.inputs(), true).rand();
	
	RandomEngine::seed(0);
	auto &o1 = m1.forward(input);
	
	RandomEngine::seed(0);
	auto &o2 = m2.forward(input);
	
	for(auto x = o1.begin(), y = o2.begin(), end = o1.end(); x != end; ++x, ++y)
	{
		if(std::fabs(*x - *y) > 1e-12)
			return false;
	}
	
	return true;
}

template <typename T>
bool TestModule_flattening(T &module)
{
	if(module.parameterList().size() > 0)
	{
		// flatten tensors
		auto &old = module.parameters();
		
		// intentionally break shared connection
		*module.parameterList()[0] = module.parameterList()[0]->copy();
		
		// intentionally add an extra shared connection
		auto view = old.view(0);
		
		// reflatten
		module.parameters();
		
		// ensure shared connection is back
		if(!module.parameterList()[0]->sharedWith(module.parameters()))
			return false;
	}
	
	if(module.gradList().size() > 0)
	{
		// flatten tensors
		auto &old = module.grad();
		
		// intentionally break shared connection
		*module.gradList()[0] = module.gradList()[0]->copy();
		
		// intentionally add an extra shared connection
		auto view = old.view(0);
		
		// reflatten
		module.grad();
		
		// ensure shared connection is back
		if(!module.gradList()[0]->sharedWith(module.grad()))
			return false;
	}
	
	if(module.stateList().size() > 0)
	{
		// flatten tensors
		auto &old = module.state();
		
		// intentionally break shared connection
		*module.stateList()[0] = module.stateList()[0]->copy();
		
		// intentionally add an extra shared connection
		auto view = old.view(0);
		
		// reflatten
		module.state();
		
		// ensure shared connection is back
		if(!module.stateList()[0]->sharedWith(module.state()))
			return false;
	}
	
	return true;
}

template <typename T>
void TestModule_copyConstructor(T &module)
{
	module.parameters().rand();
	T copy(module);
	NNAssert(TestModule_equalParams(module, copy), "Module::Module(const Module &) failed! Parameters are not equal!");
	NNAssert(TestModule_notShared(module, copy), "Module::Module(const Module &) failed! Sharing parameters; not a deep copy!");
	NNAssert(TestModule_equalOutput(module, copy), "Module::Module(const Module &) failed! Different outputs for the same input!");
}

template <typename T>
void TestModule_assignment(T &module)
{
	module.parameters().rand();
	T copy;
	copy = module;
	NNAssert(TestModule_equalParams(module, copy), "Module::operator=(const Module &) failed! Parameters are not equal!");
	NNAssert(TestModule_notShared(module, copy), "Module::operator=(const Module &) failed! Sharing parameters; not a deep copy!");
	NNAssert(TestModule_equalOutput(module, copy), "Module::operator=(const Module &) failed! Different outputs for the same input!");
}

template <typename T>
void TestModule_serialization(T &module)
{
	T s1 = Serialized(module).as<T>();
	NNAssert(TestModule_equalParams(module, s1), "Serialization through reference failed! Parameters are not equal!");
	NNAssert(TestModule_equalOutput(module, s1), "Serialization through reference failed! Different outputs for the same input!");
	
	T s2 = Serialized(&module).as<T>();
	NNAssert(TestModule_equalParams(module, s2), "Serialization through pointer failed! Parameters are not equal!");
	NNAssert(TestModule_equalOutput(module, s2), "Serialization through pointer failed! Different outputs for the same input!");
	
	Module<> *s3 = Serialized(module).as<Module<> *>();
	NNAssert(TestModule_equalParams(module, *s3), "Generic serialization through reference failed! Parameters are not equal!");
	NNAssert(TestModule_equalOutput(module, *s3), "Generic serialization through reference failed! Different outputs for the same input!");
	delete s3;
	
	Module<> *s4 = Serialized(&module).as<Module<> *>();
	NNAssert(TestModule_equalParams(module, *s4), "Generic serialization through pointer failed! Parameters are not equal!");
	NNAssert(TestModule_equalOutput(module, *s4), "Generic serialization through pointer failed! Different outputs for the same input!");
	delete s4;
}

template <typename T>
void TestModule(T &module)
{
	TestModule_copyConstructor(module);
	TestModule_assignment(module);
	TestModule_serialization(module);
	TestModule_flattening(module);
}

#endif
