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

            RandomEngine::sharedEngine().seed(0);
            auto input = math::rand(Tensor<T>(nnImpl.inputShape(), true));
            auto output = math::rand(Tensor<T>(nnImpl.outputShape(), true));

            auto copy = Serialized(nnImpl).get<Module<T> *>();
            NNTestEquals(nnImpl.inputShape(), copy->inputShape());
            NNTestEquals(nnImpl.outputShape(), copy->outputShape());
            forEach([&](T origParam, T copyParam)
            {
                NNTestAlmostEquals(origParam, copyParam, 1e-12);
            }, nnImpl.params(), copy->params());

            RandomEngine::sharedEngine().seed(0);
            nnImpl.forget();
            nnImpl.grad().fill(0);
            nnImpl.forward(input);
            nnImpl.backward(input, output);

            RandomEngine::sharedEngine().seed(0);
            copy->forget();
            copy->grad().fill(0);
            copy->forward(input);
            copy->backward(input, output);

            forEach([&](T origOutput, T copyOutput)
            {
                NNTestAlmostEquals(origOutput, copyOutput, 1e-12);
            }, nnImpl.output(), copy->output());

            forEach([&](T origInGrad, T copyInGrad)
            {
                NNTestAlmostEquals(origInGrad, copyInGrad, 1e-12);
            }, nnImpl.inGrad(), copy->inGrad());

            forEach([&](T origParamGrad, T copyParamGrad)
            {
                NNTestAlmostEquals(origParamGrad, copyParamGrad, 1e-12);
            }, nnImpl.grad(), copy->grad());

            delete copy;
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
