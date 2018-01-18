#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestAbstractClassImpl(Module, Module<T>)
{
    NNTestMethod(copy)
    {
        NNTestParams()
        {
            RandomEngine::sharedEngine().seed(0);
            auto input = math::rand(Tensor<T>(nnImpl.inputShape(), true));
            auto output = math::rand(Tensor<T>(nnImpl.outputShape(), true));

            auto copy = nnImpl.copy();
            NNTestEquals(nnImpl.inputShape(), copy->inputShape());
            NNTestEquals(nnImpl.outputShape(), copy->outputShape());
            forEach([&](T origParam, T copyParam)
            {
                NNTestAlmostEquals(origParam, copyParam, 1e-12);
            }, nnImpl.params(), copy->params());

            RandomEngine::sharedEngine().seed(0);
            auto out1 = nnImpl.forward(input);
            auto ing1 = nnImpl.backward(input, output);
            auto prg1 = nnImpl.grad();

            RandomEngine::sharedEngine().seed(0);
            auto out2 = copy->forward(input);
            auto ing2 = copy->backward(input, output);
            auto prg2 = copy->grad();

            forEach([&](T origOutput, T copyOutput)
            {
                NNTestAlmostEquals(origOutput, copyOutput, 1e-12);
            }, out1, out2);

            forEach([&](T origInGrad, T copyInGrad)
            {
                NNTestAlmostEquals(origInGrad, copyInGrad, 1e-12);
            }, ing1, ing2);

            forEach([&](T origParamGrad, T copyParamGrad)
            {
                NNTestAlmostEquals(origParamGrad, copyParamGrad, 1e-12);
            }, prg1, prg2);

            delete copy;
        }
    }

    NNTestMethod(forget)
    {
        NNTestParams()
        {
            nnImpl.state().fill(1);
            auto oldState = nnImpl.state().copy();
            nnImpl.forget();
            forEach([&](T oldState, T newState)
            {
                NNTestAlmostEquals(oldState, 1, 1e-12);
                NNTestAlmostEquals(newState, 0, 1e-12);
            }, oldState, nnImpl.state());
        }
    }

    NNTestMethod(save)
    {
        NNTestParams(Serialized &)
        {
            Serialized s;
            nnImpl.save(s);
            NNTestEquals(nnImpl.inputShape(), s.get<Storage<size_t>>("inputShape"));
            NNTestEquals(nnImpl.outputShape(), s.get<Storage<size_t>>("outputShape"));
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);
            auto input = math::rand(Tensor<T>(nnImpl.inputShape(), true));

            RandomEngine::sharedEngine().seed(0);
            nnImpl.forget();
            auto out1 = nnImpl.forward(input).copy();

            RandomEngine::sharedEngine().seed(0);
            nnImpl.forget();
            auto out2 = nnImpl.forward(input);

            forEach([&](T output1, T output2)
            {
                NNTestAlmostEquals(output1, output2, 1e-12);
            }, out1, out2);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);
            auto input = math::rand(Tensor<T>(nnImpl.inputShape(), true));
            auto output = math::rand(Tensor<T>(nnImpl.outputShape(), true));

            RandomEngine::sharedEngine().seed(0);
            nnImpl.forget();
            nnImpl.grad().fill(0);
            nnImpl.forward(input);
            auto ing1 = nnImpl.backward(input, output).copy();
            auto prg1 = nnImpl.grad().copy();

            RandomEngine::sharedEngine().seed(0);
            nnImpl.forget();
            nnImpl.grad().fill(0);
            nnImpl.forward(input);
            auto ing2 = nnImpl.backward(input, output);
            auto prg2 = nnImpl.grad();

            forEach([&](T inGrad1, T inGrad2)
            {
                NNTestAlmostEquals(inGrad1, inGrad2, 1e-12);
            }, ing1, ing2);

            forEach([&](T paramGrad1, T paramGrad2)
            {
                NNTestAlmostEquals(paramGrad1, paramGrad2, 1e-12);
            }, prg1, prg2);
        }
    }

    NNTestMethod(paramsList)
    {
        NNTestParams()
        {
            if(nnImpl.paramsList().size() > 0)
            {
                // flatten tensors
                auto &old = nnImpl.params();

                // intentionally break shared connection
                *nnImpl.paramsList()[0] = nnImpl.paramsList()[0]->copy();

                // intentionally add an extra shared connection
                auto view = old.view(0);

                // reflatten
                nnImpl.params();

                // ensure shared connection is back
                NNTest(nnImpl.paramsList()[0]->sharedWith(nnImpl.params()));
            }
        }
    }

    NNTestMethod(gradList)
    {
        NNTestParams()
        {
            if(nnImpl.gradList().size() > 0)
            {
                // flatten tensors
                auto &old = nnImpl.grad();

                // intentionally break shared connection
                *nnImpl.gradList()[0] = nnImpl.gradList()[0]->copy();

                // intentionally add an extra shared connection
                auto view = old.view(0);

                // reflatten
                nnImpl.grad();

                // ensure shared connection is back
                NNTest(nnImpl.gradList()[0]->sharedWith(nnImpl.grad()));
            }
        }
    }

    NNTestMethod(stateList)
    {
        NNTestParams()
        {
            if(nnImpl.stateList().size() > 0)
            {
                // flatten tensors
                auto &old = nnImpl.state();

                // intentionally break shared connection
                *nnImpl.stateList()[0] = nnImpl.stateList()[0]->copy();

                // intentionally add an extra shared connection
                auto view = old.view(0);

                // reflatten
                nnImpl.state();

                // ensure shared connection is back
                NNTest(nnImpl.stateList()[0]->sharedWith(nnImpl.state()));
            }
        }
    }

    NNTestMethod(params)
    {
        NNTestParams()
        {
            Tensor<T> p1 = nnImpl.params();
            Tensor<T> p2 = nnImpl.params();
            NNTest(p1.sharedWith(p2));
        }
    }

    NNTestMethod(grad)
    {
        NNTestParams()
        {
            Tensor<T> g1 = nnImpl.params();
            Tensor<T> g2 = nnImpl.params();
            NNTest(g1.sharedWith(g2));
        }
    }

    NNTestMethod(state)
    {
        NNTestParams()
        {
            Tensor<T> s1 = nnImpl.params();
            Tensor<T> s2 = nnImpl.params();
            NNTest(s1.sharedWith(s2));
        }
    }

    NNTestMethod(output)
    {
        NNTestParams()
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> *out1 = &nnImpl.forward(math::rand(Tensor<T>(nnImpl.inputShape(), true)));
            Tensor<T> *out2 = &nnImpl.output();
            NNTestEquals(out1, out2);
        }
    }

    NNTestMethod(inGrad)
    {
        NNTestParams()
        {
            RandomEngine::sharedEngine().seed(0);
            auto input = math::rand(Tensor<T>(nnImpl.inputShape(), true));
            auto output = math::rand(Tensor<T>(nnImpl.outputShape(), true));
            nnImpl.forward(input);
            Tensor<T> *ing1 = &nnImpl.backward(input, output);
            Tensor<T> *ing2 = &nnImpl.inGrad();
            NNTestEquals(ing1, ing2);
        }
    }
}

/*
template <template <typename> class M, typename T = NN_REAL_T>
class ModuleTests
{
public:
    static void run(const std::string &name, M<T> &module, const Tensor<T> &sampleInput, bool randomizeInput = true)
    {
        testDeterministic(name, module, randomizeInput ? math::rand(Tensor<T>(sampleInput.shape(), true)) : sampleInput);
        testState(name, module, randomizeInput ? math::rand(Tensor<T>(sampleInput.shape(), true)) : sampleInput);
        testCopyConstructor(name, module, randomizeInput ? math::rand(Tensor<T>(sampleInput.shape(), true)) : sampleInput);
        testAssignment(name, module, randomizeInput ? math::rand(Tensor<T>(sampleInput.shape(), true)) : sampleInput);
        testSerialization(name, module, randomizeInput ? math::rand(Tensor<T>(sampleInput.shape(), true)) : sampleInput);
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
        math::rand(module.params());
        M<T> copy(module);
        NNAssert(testEqualParams(module, copy), name + "::" + name + "(const " + name + " &) failed! Parameters are not equal!");
        NNAssert(testNotShared(module, copy), name + "::" + name + "(const " + name + " &) failed! Sharing parameters; not a deep copy!");
        NNAssert(testEqualOutput(module, copy, input), name + "::" + name + "(const " + name + " &) failed! Different outputs for the same input!");
    }

    static void testAssignment(const std::string &name, M<T> &module, const Tensor<T> &input)
    {
        math::rand(module.params());

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
*/
