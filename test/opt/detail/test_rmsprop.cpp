#include "../test_optimizer.hpp"
#include "../test_rmsprop.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/opt/rmsprop.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(RMSProp)
{
    Linear<T> model(2, 3);
    NNRunAbstractTest(Optimizer, RMSProp, new RMSProp<T>(model));

    NNTestMethod(gamma)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            RMSProp<T> opt(model);
            opt.learningRate(0.25);
            opt.gamma(0.25);
            NNTestAlmostEquals(opt.gamma(), 0.25, 1e-12);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.71132486707, 1e-9);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.47515671489, 1e-9);
        }
    }

    NNTestMethod(step)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Linear<T> model(1, 1, false);
            model.params().copy({ 1 });

            RMSProp<T> opt(model);
            opt.learningRate(0.25);
            opt.gamma(0);

            Tensor<T> inputs = { 1 };
            Tensor<T> target = { 0 };

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.75000000125, 1e-9);

            opt.step(inputs, target);
            NNTestAlmostEquals(opt.params()(0), 0.50000000291, 1e-9);
        }
    }
}
