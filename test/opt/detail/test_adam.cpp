#include "../test_adam.hpp"
#include "nnlib/opt/adam.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/critics/mse.hpp"
#include "nnlib/util/random.hpp"
using namespace nnlib;

void TestAdam()
{
	RandomEngine::sharedEngine().seed(0);

	Tensor<NN_REAL_T> feat = Tensor<NN_REAL_T>(10, 2).rand();
	Tensor<NN_REAL_T> lab = Tensor<NN_REAL_T>(10, 3).rand();

	Linear<NN_REAL_T> nn(2, 3);
	MSE<NN_REAL_T> critic;

	double errBefore = critic.forward(nn.forward(feat), lab);

	Adam<NN_REAL_T> opt(nn, critic);
	opt.step(feat.narrow(0, 0), lab.narrow(0, 0));
	opt.step(feat, lab);

	double errAfter = critic.forward(nn.forward(feat), lab);

	NNAssertLessThan(errAfter - errBefore, 0, "Optimization failed!");
}
