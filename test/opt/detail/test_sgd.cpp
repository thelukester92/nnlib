#include "../test_sgd.hpp"
#include "nnlib/critics/mse.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/opt/sgd.hpp"
using namespace nnlib;

void TestSGD()
{
    RandomEngine::sharedEngine().seed(0);

    Tensor<NN_REAL_T> feat = math::rand(Tensor<NN_REAL_T>(10, 2));
    Tensor<NN_REAL_T> lab = math::rand(Tensor<NN_REAL_T>(10, 3));

    Linear<NN_REAL_T> nn(2, 3);
    MSE<NN_REAL_T> critic;

    double errBefore = critic.forward(nn.forward(feat), lab);

    SGD<NN_REAL_T> opt(nn, critic);
    opt.step(feat.narrow(0, 0), lab.narrow(0, 0));
    opt.step(feat, lab);

    double errAfter = critic.forward(nn.forward(feat), lab);

    NNAssertLessThan(errAfter - errBefore, 0, "Optimization failed!");

    errBefore = errAfter;
    opt.momentum(0.25);
    opt.step(feat, lab);
    errAfter = critic.forward(nn.forward(feat), lab);

    NNAssertLessThan(errAfter - errBefore, 0, "Optimization failed!");
}
