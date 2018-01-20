#include "../test_nadam.hpp"
#include "../test_optimizer.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/opt/nadam.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Nadam)
{
    Linear<T> model(2, 3);
    NNRunAbstractTest(Optimizer, Nadam, new Nadam<T>(model));

    NNTestMethod(beta1)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            Nadam<T> opt(model);
            opt.learningRate(0.25);
            opt.beta1(0.25);
            opt.beta2(0);
            NNTestAlmostEquals(opt.beta1(), 0.25, 1e-12);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.6875000015624999, 1e-12);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.4193181853724172, 1e-12);
        }
    }

    NNTestMethod(beta2)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            Nadam<T> opt(model);
            opt.learningRate(0.25);
            opt.beta1(0);
            opt.beta2(0.25);
            NNTestAlmostEquals(opt.beta2(), 0.25, 1e-12);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.7500000014433756, 1e-12);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.5174348754405044, 1e-12);
        }
    }

    NNTestMethod(step)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            Nadam<T> opt(model);
            opt.learningRate(0.25);
            opt.beta1(0);
            opt.beta2(0);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.75000000125, 1e-9);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.5000000029166667, 1e-9);
        }
    }
}
