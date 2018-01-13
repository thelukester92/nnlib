#include "../test_dropconnect.hpp"
#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/dropconnect.hpp"
#include "nnlib/nn/linear.hpp"
using namespace nnlib;

void TestDropConnect()
{
    RandomEngine::sharedEngine().seed(0);

    Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>(4, 25).ones();
    Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>(4, 1).ones();
    double p = 0.75, sum1 = 0, sum2 = 0;
    size_t c = 100;

    Linear<NN_REAL_T> *linear = new Linear<NN_REAL_T>(25, 1);
    linear->weights().ones();
    linear->bias().zeros();

    DropConnect<NN_REAL_T> module(linear, p);
    NNAssertEquals(module.dropProbability(), p, "DropConnect::DropConnect failed!");

    for(size_t i = 0; i < c; ++i)
    {
        sum1 += math::sum(module.forward(inp)) / c;
        sum2 += math::sum(module.backward(inp, grd)) / c;
    }

    NNAssertAlmostEquals(sum1, c * (1 - p), 2, "DropConnect::forward failed! (Note: this may be due to the random process).");
    NNAssertEquals(sum1, sum2, "DropConnect::backward failed!");

    module.dropProbability(p = 0.5).training(false);
    sum1 = sum2 = 0;

    for(size_t i = 0; i < c; ++i)
    {
        sum1 += math::sum(module.forward(inp)) / c;
        sum2 += math::sum(module.backward(inp, grd)) / c;
    }

    NNAssertAlmostEquals(sum1, c * (1 - p), 2, "DropConnect::forward (inference) failed! (Note: this may be due to the random process).");
    NNAssertEquals(sum1, sum2, "DropConnect::backward (inference) failed!");

    module.state().fill(0);
    NNAssertAlmostEquals(math::sum(module.output()), 0, 1e-12, "DropConnect::state failed!");

    {
        auto l1 = module.paramsList();
        auto l2 = linear->paramsList();
        NNAssertEquals(l1.size(), l2.size(), "DropConnect::paramsList failed!");
        for(auto i = l1.begin(), j = l2.begin(), end = l1.end(); i != end; ++i, ++j)
            NNAssertEquals(*i, *j, "DropConnect::paramsList failed!");
    }

    {
        auto l1 = module.gradList();
        auto l2 = linear->gradList();
        NNAssertEquals(l1.size(), l2.size(), "DropConnect::gradList failed!");
        for(auto i = l1.begin(), j = l2.begin(), end = l1.end(); i != end; ++i, ++j)
            NNAssertEquals(*i, *j, "DropConnect::gradList failed!");
    }

    TestModule("DropConnect", module, inp);
}
