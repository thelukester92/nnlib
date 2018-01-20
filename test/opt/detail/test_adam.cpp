#include "../test_adam.hpp"
#include "../test_optimizer.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/opt/adam.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Adam)
{
    Linear<T> model(2, 3);
    NNRunAbstractTest(Optimizer, Adam, new Adam<T>(model));

    NNTestMethod(beta1)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            Adam<T> opt(model);
            opt.learningRate(0.25);
            opt.beta1(0.25);
            opt.beta2(0);
            NNTestAlmostEquals(opt.beta1(), 0.25, 1e-12);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.75000000125, 1e-9);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.48333333647, 1e-9);
        }
    }

    NNTestMethod(beta2)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            Adam<T> opt(model);
            opt.learningRate(0.25);
            opt.beta1(0);
            opt.beta2(0.25);
            NNTestAlmostEquals(opt.beta2(), 0.25, 1e-12);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.75000000144, 1e-9);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.51743487543, 1e-9);
        }
    }

    NNTestMethod(step)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            Adam<T> opt(model);
            opt.learningRate(0.25);
            opt.beta1(0);
            opt.beta2(0);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.75000000125, 1e-9);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.50000000291, 1e-9);
        }
    }
}
