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
}
