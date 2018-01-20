#include "../test_optimizer.hpp"
#include "../test_sgd.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/opt/sgd.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(SGD)
{
    Linear<T> model(2, 3);
    NNRunAbstractTest(Optimizer, SGD, new SGD<T>(model));

    NNTestMethod(momentum)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            SGD<T> opt(model);
            opt.learningRate(0.25);
            opt.momentum(0.5);
            NNTestAlmostEquals(opt.momentum(), 0.5, 1e-12);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.25, 1e-12);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), -0.0625, 1e-12);
        }
    }

    NNTestMethod(step)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            SGD<T> opt(model);
            opt.learningRate(0.25);
            opt.momentum(0);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.5, 1e-12);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.25, 1e-12);
        }
    }
}
