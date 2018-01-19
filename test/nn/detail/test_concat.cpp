#include "../test_concat.hpp"
#include "../test_container.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/concat.hpp"
#include "nnlib/nn/linear.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Concat)
{
    NNRunAbstractTest(Container, Concat, new Concat<T>(new Linear<T>(3, 4), new Linear<T>(3, 2)));

    NNTestMethod(Concat)
    {
        NNTestParams(Module *, Module *)
        {
            Linear<T> *comp1 = new Linear<T>(3, 4);
            Linear<T> *comp2 = new Linear<T>(3, 2);
            Concat<T> module(comp1, comp2);
            NNTestEquals(module.inputShape()[1], 3);
            NNTestEquals(module.outputShape()[1], 6);
            NNTestEquals(module.component(0), comp1);
            NNTestEquals(module.component(1), comp2);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Concat &)
        {
            Concat<T> orig(new Linear<T>(3, 4), new Linear<T>(3, 2));
            Concat<T> copy(new Linear<T>(3, 4), new Linear<T>(3, 2));
            copy = orig;
            NNTestEquals(orig.components(), copy.components());
            forEach([&](T orig, T copy)
            {
                NNTestAlmostEquals(orig, copy, 1e-12);
            }, orig.params(), copy.params());
        }
    }

    NNTestMethod(concatDim)
    {
        NNTestParams()
        {
            Concat<T> module(new Linear<T>(3, 4), new Linear<T>(3, 4));
            NNTestEquals(module.concatDim(), 1);
            module.concatDim(0);
            NNTestEquals(module.concatDim(), 0);
        }

        NNTestParams(size_t)
        {
            Concat<T> module(new Linear<T>(3, 4), new Linear<T>(3, 4));
            module.concatDim(0);
            NNTestEquals(module.outputShape()[0], 2);
            NNTestEquals(module.outputShape()[1], 4);
        }
    }

    NNTestMethod(save)
    {
        NNTestParams(Serialized &)
        {
            Concat<T> module(new Linear<T>(3, 4), new Linear<T>(3, 4));
            Serialized s;
            module.save(s);
            NNTestEquals(s.get<size_t>("concatDim"), 1);
            module.concatDim(0);
            module.save(s);
            NNTestEquals(s.get<size_t>("concatDim"), 0);
        }
    }

    NNTestMethod(add)
    {
        NNTestParams(Module *)
        {
            Concat<T> module(new Linear<T>(3, 4), new Linear<T>(3, 2));
            module.add(new Linear<T>(3, 12));
            NNTestEquals(module.components(), 3);
            NNTestEquals(module.outputShape()[1], 18);
        }
    }

    NNTestMethod(remove)
    {
        NNTestParams(size_t)
        {
            auto comp1 = new Linear<T>(3, 4);
            auto comp2 = new Linear<T>(3, 2);
            auto comp3 = new Linear<T>(3, 12);
            auto comp4 = new Linear<T>(3, 2);
            Concat<T> module(comp1, comp2, comp3, comp4);
            NNTestEquals(module.remove(1), comp2);
            NNTestEquals(module.components(), 3);
            NNTestEquals(module.outputShape()[1], 18);
            NNTestEquals(module.remove(2), comp4);
            NNTestEquals(module.components(), 2);
            NNTestEquals(module.outputShape()[1], 16);
            NNTestEquals(module.remove(0), comp1);
            NNTestEquals(module.components(), 1);
            NNTestEquals(module.outputShape()[1], 12);
            NNTestEquals(module.remove(0), comp3);
            NNTestEquals(module.components(), 0);
            delete comp1;
            delete comp2;
            delete comp3;
            delete comp4;
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            auto comp1 = new Linear<T>(2, 2, false);
            auto comp2 = new Linear<T>(2, 3, false);
            comp1->params().copy({ 0, 1, 2, 3 });
            comp2->params().copy({ 4, 5, 6, 7, 8, 9 });

            Tensor<T> input({ 0.5, 2 });
            Tensor<T> target({ 4, 6.5, 16, 18.5, 21 });
            Concat<T> module(comp1, comp2);
            module.forward(input);

            forEach([&](T output, T target)
            {
                NNTestAlmostEquals(output, target, 1e-12);
            }, module.output(), target);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &)
        {
            auto comp1 = new Linear<T>(2, 2, false);
            auto comp2 = new Linear<T>(2, 3, false);

            Concat<T> module(comp1, comp2);
            module.params().copy({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            Tensor<T> input({ 0.5, 2 });
            Tensor<T> grad({ 0.5, 0, 1, 2, -1 });
            Tensor<T> inGrad({ 8, 15 });
            Tensor<T> pGrad({ 0.25, 0, 1, 0, 0.5, 1, -0.5, 2, 4, -2 });

            module.forward(input);
            module.backward(input, grad);

            forEach([&](T inGrad, T target)
            {
                NNTestAlmostEquals(inGrad, target, 1e-12);
            }, module.inGrad(), inGrad);

            forEach([&](T pGrad, T target)
            {
                NNTestAlmostEquals(pGrad, target, 1e-12);
            }, module.grad(), pGrad);
        }
    }
}
