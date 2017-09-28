#ifndef TEST_MODULE_H
#define TEST_MODULE_H

#include "nnlib/nn/module.h"
using namespace std;
using namespace nnlib;

template <template <typename> class M, typename T = double>
class ModuleTests
{
public:
	static void run(const std::string &name, M<T> &module)
	{
		testCopyConstructor(name, module);
		testAssignment(name, module);
		testSerialization(name, module);
		testFlattening(name, module);
	}
	
private:
	static void testCopyConstructor(const std::string &name, M<T> &module)
	{
		module.params().rand();
		M<T> copy(module);
		NNAssert(testEqualParams(module, copy), name + "::" + name + "(const " + name + " &) failed! Parameters are not equal!");
		NNAssert(testNotShared(module, copy), name + "::" + name + "(const " + name + " &) failed! Sharing parameters; not a deep copy!");
		NNAssert(testEqualOutput(module, copy), name + "::" + name + "(const " + name + " &) failed! Different outputs for the same input!");
	}
	
	static void testAssignment(const std::string &name, M<T> &module)
	{
		module.params().rand();
		
		M<T> *copy = static_cast<M<T> *>(module.copy());
		copy->params().fill(0);
		*copy = module;
		
		NNAssert(testEqualParams(module, *copy), name + "::operator=(const " + name + " &) failed! Parameters are not equal!");
		NNAssert(testNotShared(module, *copy), name + "::operator=(const " + name + " &) failed! Sharing parameters; not a deep copy!");
		NNAssert(testEqualOutput(module, *copy), name + "::operator=(const " + name + " &) failed! Different outputs for the same input!");
		
		delete copy;
	}
	
	static void testSerialization(const std::string &name, M<T> &module)
	{
		M<T> s1 = Serialized(module).as<M<T>>();
		NNAssert(testEqualParams(module, s1), "Serialization through reference failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, s1), "Serialization through reference failed! Different outputs for the same input!");
		
		M<T> s2 = Serialized(&module).as<M<T>>();
		NNAssert(testEqualParams(module, s2), "Serialization through pointer failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, s2), "Serialization through pointer failed! Different outputs for the same input!");
		
		Module<T> *s3 = Serialized(module).as<Module<T> *>();
		NNAssert(testEqualParams(module, *s3), "Generic serialization through reference failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, *s3), "Generic serialization through reference failed! Different outputs for the same input!");
		delete s3;
		
		Module<T> *s4 = Serialized(&module).as<Module<T> *>();
		NNAssert(testEqualParams(module, *s4), "Generic serialization through pointer failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, *s4), "Generic serialization through pointer failed! Different outputs for the same input!");
		delete s4;
	}
	
	static void testFlattening(const std::string &name, M<T> &module)
	{
		if(module.paramsList().size() > 0)
		{
			// flatten tensors
			auto &old = module.params();
			
			// intentionally break shared connection
			*module.paramsList()[0] = module.paramsList()[0]->copy();
			
			// intentionally add an extra shared connection
			auto view = old.view(0);
			
			// reflatten
			module.params();
			
			// ensure shared connection is back
			NNAssert(module.paramsList()[0]->sharedWith(module.params()), name + "::params() failed! Shared connections did not work correctly.");
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
			NNAssert(module.gradList()[0]->sharedWith(module.grad()), name + "::grad() failed! Shared connections did not work correctly.");
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
			NNAssert(module.stateList()[0]->sharedWith(module.state()), name + "::state() failed! Shared connections did not work correctly.");
		}
	}
	
	static bool testEqualParams(Module<T> &m1, Module<T> &m2)
	{
		auto &p1 = m1.params();
		auto &p2 = m2.params();
		
		if(p1.shape() != p2.shape())
			return false;
		
		for(auto x = p1.begin(), y = p2.begin(), end = p1.end(); x != end; ++x, ++y)
		{
			if(std::abs(*x - *y) > 1e-12)
				return false;
		}
		
		return true;
	}
	
	static bool testNotShared(Module<T> &m1, Module<T> &m2)
	{
		return !m1.params().sharedWith(m2.params());
	}
	
	static bool testEqualOutput(Module<T> &m1, Module<T> &m2)
	{
		Tensor<T> input = Tensor<T>(m1.inGrad().shape(), true).rand();
		
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
};

template <template <typename> class M, typename T>
void TestModule(const std::string &name, M<T> &module)
{
	ModuleTests<M, T>::run(name, module);
}

#endif
