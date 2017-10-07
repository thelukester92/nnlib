#ifndef TEST_CONTAINER_H
#define TEST_CONTAINER_H

#include "nnlib/nn/container.hpp"
#include "test_module.hpp"
using namespace std;
using namespace nnlib;

template <template <typename> class M, typename T = double>
class ContainerTests : public ModuleTests<M, T>
{
public:
	static void run(const std::string &name, M<T> &module, const Tensor<T> &sampleInput)
	{
		ModuleTests<M, T>::run(name, module, sampleInput);
		testSerialization(name, module, sampleInput);
	}
	
protected:
	using ModuleTests<M, T>::testEqualParams;
	using ModuleTests<M, T>::testEqualOutput;
	
private:
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
