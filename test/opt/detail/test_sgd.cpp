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
            Linear<T> model(1, 1);
            model.params().fill(1);
            SGD<T> opt(model);
            opt.momentum(0.5);
            NNTestAlmostEquals(opt.momentum(), 0.5, 1e-12);
        }
    }
}
