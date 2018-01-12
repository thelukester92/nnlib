#include "../test_mse.hpp"
#include "nnlib/critics/mse.hpp"
#include "nnlib/math/math.hpp"
using namespace nnlib;

void TestMSE()
{
	Storage<size_t> shape = { 5, 1 };
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({  1,  2,  3,  4,  5 }).resize(shape);
	Tensor<NN_REAL_T> tgt = Tensor<NN_REAL_T>({  2,  4,  6,  8,  0 }).resize(shape);
	Tensor<NN_REAL_T> sqd = Tensor<NN_REAL_T>({  1,  4,  9, 16, 25 }).resize(shape);
	Tensor<NN_REAL_T> dif = Tensor<NN_REAL_T>({ -2, -4, -6, -8, 10 }).resize(shape);
	MSE<NN_REAL_T> critic(false);

	double mse = critic.forward(inp, tgt);
	NNAssert(fabs(mse - math::sum(sqd)) < 1e-12, "MSE::forward with no average failed!");

	critic.average(true);
	mse = critic.forward(inp, tgt);
	NNAssert(fabs(mse - math::mean(sqd)) < 1e-12, "MSE::forward with average failed!");

	critic.average(false);
	critic.backward(inp, tgt);
	NNAssert(math::sum(math::square(critic.inGrad() - dif)) < 1e-12, "MSE::backward with no average failed!");

	critic.average(true);
	critic.backward(inp, tgt);
	NNAssert(math::sum(math::square(critic.inGrad() - dif / dif.size())) < 1e-12, "MSE::backward with average failed!");
}
