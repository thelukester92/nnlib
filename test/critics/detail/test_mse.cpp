#include "../test_mse.hpp"
#include "nnlib/critics/mse.hpp"
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
	NNHardAssert(fabs(mse - sqd.sum()) < 1e-12, "MSE<NN_REAL_T>::forward with no average failed!");
	
	critic.average(true);
	mse = critic.forward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.mean()) < 1e-12, "MSE<NN_REAL_T>::forward with average failed!");
	
	critic.average(false);
	critic.backward(inp, tgt);
	NNHardAssert(critic.inGrad().addM(dif, -1).square().sum() < 1e-12, "MSE<NN_REAL_T>::backward with no average failed!");
	
	critic.average(true);
	critic.backward(inp, tgt);
	NNHardAssert(critic.inGrad().addM(dif.scale(1.0 / dif.size()), -1).square().sum() < 1e-12, "MSE<NN_REAL_T>::backward with average failed!");
}
