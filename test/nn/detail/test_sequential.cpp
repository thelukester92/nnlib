#include "../test_container.hpp"
#include "../test_sequential.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/sequential.hpp"
#include "nnlib/nn/batchnorm.hpp"
#include "nnlib/nn/linear.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Sequential)
{
    NNRunAbstractTest(Container, Sequential, new Sequential<T>(new Linear<T>(3, 4), new Linear<T>(4, 2)));

    NNTestMethod(Sequential)
    {
        NNTestParams(Module *, Module *)
        {
            Linear<T> *comp1 = new Linear<T>(3, 4);
            Linear<T> *comp2 = new Linear<T>(4, 2);
            Sequential<T> module(comp1, comp2);
            NNTestEquals(module.inputShape()[1], 3);
            NNTestEquals(module.outputShape()[1], 2);
            NNTestEquals(module.component(0), comp1);
            NNTestEquals(module.component(1), comp2);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Concat &)
        {
            Sequential<T> orig(new Linear<T>(3, 4), new Linear<T>(4, 2));
            Sequential<T> copy(new Linear<T>(1, 2), new Linear<T>(2, 1));
            copy = orig;
            NNTestEquals(orig.components(), copy.components());
            forEach([&](T orig, T copy)
            {
                NNTestAlmostEquals(orig, copy, 1e-12);
            }, orig.params(), copy.params());
        }
    }

    NNTestMethod(add)
    {
        NNTestParams(Module *)
        {
            Sequential<T> module(new Linear<T>(3, 4), new Linear<T>(4, 2));
            module.add(new Linear<T>(2, 12));
            NNTestEquals(module.components(), 3);
            NNTestEquals(module.outputShape()[1], 12);
        }
    }

    NNTestMethod(remove)
    {
        NNTestParams(size_t)
        {
            auto comp1 = new Linear<T>(3, 4);
            auto comp2 = new Linear<T>(4, 2);
            auto comp3 = new Linear<T>(2, 4);
            auto comp4 = new Linear<T>(4, 5);
            auto comp5 = new Linear<T>(5, 10);
            Sequential<T> module(comp1, comp2, comp3, comp4, comp5);
            NNTestEquals(module.remove(1), comp2);
            NNTestEquals(module.remove(1), comp3);
            NNTestEquals(module.components(), 3);
            NNTestEquals(module.inputShape()[1], 3);
            NNTestEquals(module.outputShape()[1], 10);
            NNTestEquals(module.remove(2), comp5);
            NNTestEquals(module.components(), 2);
            NNTestEquals(module.inputShape()[1], 3);
            NNTestEquals(module.outputShape()[1], 5);
            NNTestEquals(module.remove(0), comp1);
            NNTestEquals(module.components(), 1);
            NNTestEquals(module.inputShape()[1], 4);
            NNTestEquals(module.outputShape()[1], 5);
            NNTestEquals(module.remove(0), comp4);
            NNTestEquals(module.components(), 0);
            delete comp1;
            delete comp2;
            delete comp3;
            delete comp4;
            delete comp5;
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
            Tensor<T> target({ 61.5, 72, 82.5 });
            Sequential<T> module(comp1, comp2);
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

            Sequential<T> module(comp1, comp2);
            module.params().copy({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            Tensor<T> input({ 0.5, 2 });
            Tensor<T> grad({ 0.5, 0, -1 });
            Tensor<T> inGrad({ -5.5, -24.5 });
            Tensor<T> pGrad({ -2, -2.75, -8, -11, 2, 0, -4, 3.25, 0, -6.5 });

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
