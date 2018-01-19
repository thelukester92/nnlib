#include "../test_adam.hpp"
#include "../test_optimizer.hpp"
#include "nnlib/critics/mse.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/opt/adam.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Adam)
{
    Linear<T> model(2, 3);
    MSE<T> critic;
    NNRunAbstractTest(Optimizer, Adam, new Adam<T>(model, critic));
}
