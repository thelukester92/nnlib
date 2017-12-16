#ifndef TEST_MAP_HPP
#define TEST_MAP_HPP

#include "nnlib/nn/map.hpp"
#include "test_module.hpp"
using namespace std;
using namespace nnlib;

template <template <typename> class M, typename T = NN_REAL_T>
class MapTests : public ModuleTests<M, T>
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
	static void testSerialization(const std::string &name, Map<T> &module, const Tensor<T> &sampleInput)
	{
		Map<T> *s1 = Serialized(module).as<Map<T> *>();
		NNAssert(testEqualParams(module, *s1), "Serialization through Map reference failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, *s1, sampleInput), "Serialization through Map reference failed! Different outputs for the same input!");
		delete s1;
		
		Map<T> *s2 = Serialized(&module).as<Map<T> *>();
		NNAssert(testEqualParams(module, *s2), "Serialization through Map pointer failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, *s2, sampleInput), "Serialization through Map pointer failed! Different outputs for the same input!");
		delete s2;
	}
};

template <template <typename> class M, typename T>
void TestMap(const std::string &name, M<T> &module, const Tensor<T> &sampleInput)
{
	MapTests<M, T>::run(name, module, sampleInput);
}

#endif
