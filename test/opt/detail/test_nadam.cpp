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
    NNRunAbstractTest(Optimizer, Adam, new Nadam<T>(model));

    NNTestMethod(beta1)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1);
            model.params().fill(1);
            Nadam<T> opt(model);
            opt.beta1(0.5);
            NNTestAlmostEquals(opt.beta1(), 0.5, 1e-12);
        }
    }

    NNTestMethod(beta2)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1);
            model.params().fill(1);
            Nadam<T> opt(model);
            opt.beta2(0.5);
            NNTestAlmostEquals(opt.beta2(), 0.5, 1e-12);
        }
    }
}
