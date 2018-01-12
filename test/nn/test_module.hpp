#ifndef TEST_MODULE_HPP
#define TEST_MODULE_HPP

#include "nnlib/nn/module.hpp"
using namespace std;
using namespace nnlib;

template <template <typename> class M, typename T = NN_REAL_T>
class ModuleTests
{
public:
	static void run(const std::string &name, M<T> &module, const Tensor<T> &sampleInput, bool randomizeInput = true)
	{
		testDeterministic(name, module, randomizeInput ? Tensor<T>(sampleInput.shape(), true).rand() : sampleInput);
		testState(name, module, randomizeInput ? Tensor<T>(sampleInput.shape(), true).rand() : sampleInput);
		testCopyConstructor(name, module, randomizeInput ? Tensor<T>(sampleInput.shape(), true).rand() : sampleInput);
		testAssignment(name, module, randomizeInput ? Tensor<T>(sampleInput.shape(), true).rand() : sampleInput);
		testSerialization(name, module, randomizeInput ? Tensor<T>(sampleInput.shape(), true).rand() : sampleInput);
		testFlattening(name, module);
	}

protected:
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

	static bool testEqualOutput(Module<T> &m1, Module<T> &m2, const Tensor<T> &input)
	{
		RandomEngine::sharedEngine().seed(0);
		m1.forget();
		auto &o1 = m1.forward(input);

		RandomEngine::sharedEngine().seed(0);
		m2.forget();
		auto &o2 = m2.forward(input);

		for(auto x = o1.begin(), y = o2.begin(), end = o1.end(); x != end; ++x, ++y)
		{
			if(std::fabs(*x - *y) > 1e-12)
				return false;
		}

		return true;
	}

private:
	static void testDeterministic(const std::string &name, M<T> &module, const Tensor<T> &input)
	{
		RandomEngine::sharedEngine().seed(0);
		module.forget();
		module.grad().zeros();
		auto o1 = module.forward(input).copy();
		auto i1 = module.backward(input, module.output()).copy();
		auto p1 = module.grad().copy();

		RandomEngine::sharedEngine().seed(0);
		module.forget();
		module.grad().zeros();
		auto o2 = module.forward(input).copy();
		auto i2 = module.backward(input, module.output());
		auto p2 = module.grad();

		for(auto x = o1.begin(), y = o2.begin(), end = o1.end(); x != end; ++x, ++y)
			NNAssertAlmostEquals(*x, *y, 1e-12, name + "::forward() failed! Different outputs for the same input and random seed!");

		for(auto x = i1.begin(), y = i2.begin(), end = i1.end(); x != end; ++x, ++y)
			NNAssertAlmostEquals(*x, *y, 1e-12, name + "::backward() failed! Different input grads for the same input, outGrad, and random seed!");

		for(auto x = p1.begin(), y = p2.begin(), end = p1.end(); x != end; ++x, ++y)
			NNAssertAlmostEquals(*x, *y, 1e-12, name + "::backward() failed! Different parameter grads for the same input, outGrad, and random seed!");
	}

	static void testState(const std::string &name, M<T> &module, const Tensor<T> &input)
	{
		RandomEngine::sharedEngine().seed(0);
		auto s = module.state().copy();
		auto o1 = module.forward(input).copy();

		RandomEngine::sharedEngine().seed(0);
		module.state().copy(s);
		auto o2 = module.forward(input);

		for(auto x = o1.begin(), y = o2.begin(), end = o1.end(); x != end; ++x, ++y)
			NNAssertAlmostEquals(*x, *y, 1e-12, name + "::state() failed! Different outputs for the same state and random seed!");

		// intentionally break shared connection
		*module.stateList()[0] = module.stateList()[0]->copy();

		// forward to make unvectorized state
		module.forward(input);

		// try to clear state
		module.forget();
		for(auto x : module.output())
			NNAssertAlmostEquals(x, 0, 1e-12, name + "::forget() failed! Non-zero output!");
	}

	static void testCopyConstructor(const std::string &name, M<T> &module, const Tensor<T> &input)
	{
		module.params().rand();
		M<T> copy(module);
		NNAssert(testEqualParams(module, copy), name + "::" + name + "(const " + name + " &) failed! Parameters are not equal!");
		NNAssert(testNotShared(module, copy), name + "::" + name + "(const " + name + " &) failed! Sharing parameters; not a deep copy!");
		NNAssert(testEqualOutput(module, copy, input), name + "::" + name + "(const " + name + " &) failed! Different outputs for the same input!");
	}

	static void testAssignment(const std::string &name, M<T> &module, const Tensor<T> &input)
	{
		module.params().rand();

		M<T> copy(module);
		copy.params().fill(0);
		copy = module;

		NNAssert(testEqualParams(module, copy), name + "::operator=(const " + name + " &) failed! Parameters are not equal!");
		NNAssert(testNotShared(module, copy), name + "::operator=(const " + name + " &) failed! Sharing parameters; not a deep copy!");
		NNAssert(testEqualOutput(module, copy, input), name + "::operator=(const " + name + " &) failed! Different outputs for the same input!");
	}

	static void testSerialization(const std::string &name, M<T> &module, const Tensor<T> &input)
	{
		M<T> s1 = Serialized(module).get<M<T>>();
		NNAssert(testEqualParams(module, s1), "Serialization through reference failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, s1, input), "Serialization through reference failed! Different outputs for the same input!");

		M<T> s2 = Serialized(&module).get<M<T>>();
		NNAssert(testEqualParams(module, s2), "Serialization through pointer failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, s2, input), "Serialization through pointer failed! Different outputs for the same input!");

		Module<T> *s3 = Serialized(module).get<Module<T> *>();
		NNAssert(testEqualParams(module, *s3), "Generic serialization through reference failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, *s3, input), "Generic serialization through reference failed! Different outputs for the same input!");
		delete s3;

		Module<T> *s4 = Serialized(&module).get<Module<T> *>();
		NNAssert(testEqualParams(module, *s4), "Generic serialization through pointer failed! Parameters are not equal!");
		NNAssert(testEqualOutput(module, *s4, input), "Generic serialization through pointer failed! Different outputs for the same input!");
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
};

template <template <typename> class M, typename T>
void TestModule(const std::string &name, M<T> &module, const Tensor<T> &input, bool randomizeInput = true)
{
	ModuleTests<M, T>::run(name, module, input, randomizeInput);
}

#endif
