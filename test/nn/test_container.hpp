#ifndef TEST_CONTAINER_HPP
#define TEST_CONTAINER_HPP

#include "nnlib/nn/container.hpp"
#include "test_module.hpp"
using namespace std;
using namespace nnlib;

template <template <typename> class M, typename T = NN_REAL_T>
class ContainerTests : public ModuleTests<M, T>
{
public:
	static void run(const std::string &name, M<T> &module, const Tensor<T> &sampleInput)
	{
		ModuleTests<M, T>::run(name, module, sampleInput);
		testBuffers(name, module);
		testSerialization(name, module, sampleInput);
	}
	
protected:
	using ModuleTests<M, T>::testEqualParams;
	using ModuleTests<M, T>::testEqualOutput;
	
private:
	static void testBuffers(const std::string &name, Container<T> &module)
	{
		// make sure these have been flattened
		module.params();
		module.grad();
		module.state();
		
		// make sure they are shared
		for(size_t i = 0; i < module.components(); ++i)
		{
			NNAssert(module.component(i)->params().sharedWith(module.params()), name + "::params failed! Not shared correctly!");
			NNAssert(module.component(i)->grad().sharedWith(module.grad()), name + "::grad failed! Not shared correctly!");
			NNAssert(module.component(i)->state().sharedWith(module.state()), name + "::state failed! Not shared correctly!");
		}
		
		// make sure they are equal; this makes an assumption on the order of flattening
		auto x = module.params().begin(), y = module.grad().begin(), z = module.state().begin();
		for(size_t i = 0; i < module.components(); ++i)
		{
			for(auto xx : module.component(i)->params())
			{
				NNAssertEquals(*x, xx, name + "::params failed! Wrong value!");
				++x;
			}
			
			for(auto yy : module.component(i)->grad())
			{
				NNAssertEquals(*y, yy, name + "::grad failed! Wrong value!");
				++y;
			}
			
			for(auto zz : module.component(i)->state())
			{
				NNAssertEquals(*z, zz, name + "::state failed! Wrong value!");
				++z;
			}
		}
	}
	
	static void testSerialization(const std::string &name, Container<T> &module, const Tensor<T> &sampleInput)
	{
		Container<T> *s1 = Serialized(module).as<Container<T> *>();
		NNAssert(testEqualParams(module, *s1), "Serialization through Container reference failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, *s1, sampleInput), "Serialization through Container reference failed! Different outputs for the same input!");
		delete s1;
		
		Container<T> *s2 = Serialized(&module).as<Container<T> *>();
		NNAssert(testEqualParams(module, *s2), "Serialization through Container pointer failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, *s2, sampleInput), "Serialization through Container pointer failed! Different outputs for the same input!");
		delete s2;
	}
};

template <template <typename> class M, typename T>
void TestContainer(const std::string &name, M<T> &module, const Tensor<T> &sampleInput)
{
	ContainerTests<M, T>::run(name, module, sampleInput);
}

#endif
