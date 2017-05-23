#ifndef TEST_MSE
#define TEST_MSE

#include "nnlib/critics/mse.h"
using namespace nnlib;

void TestMSE()
{
	Storage<size_t> shape = { 5, 1 };
	Tensor<> inp = Tensor<>({  1,  2,  3,  4,  5 }).resize(shape);
	Tensor<> tgt = Tensor<>({  2,  4,  6,  8,  0 }).resize(shape);
	Tensor<> sqd = Tensor<>({  1,  4,  9, 16, 25 }).resize(shape);
	Tensor<> dif = Tensor<>({ -2, -4, -6, -8, 10 }).resize(shape);
	MSE<> critic(shape, false);
	
	double mse = critic.forward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.sum()) < 1e-12, "MSE<>::forward with no average failed!");
	
	critic.average(true);
	mse = critic.forward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.mean()) < 1e-12, "MSE<>::forward with average failed!");
	
	critic.inputs({ 10, 10 });
	mse = critic.safeForward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.mean()) < 1e-12, "MSE<>::safeForward failed!");
	
	critic.average(false);
	critic.backward(inp, tgt);
	NNHardAssert(critic.inGrad().addM(dif, -1).square().sum() < 1e-12, "MSE<>::backward failed!");
	
	critic.inputs({ 10, 10 });
	critic.safeBackward(inp, tgt);
	NNHardAssert(critic.inGrad().addM(dif, -1).square().sum() < 1e-12, "MSE<>::safeBackward failed!");
	
	critic.batch(12);
	NNHardAssert(critic.batch() == 12, "MSE<>::batch failed!");
}

#endif
