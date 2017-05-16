#include "nnlib/critics/mse.h"
#include <iostream>
using namespace nnlib;

void TestMSE()
{
	Storage<size_t> shape = { 1, 5 };
	Tensor<> inp = Tensor<>({  1,  2,  3,  4,  5 }).resize(shape);
	Tensor<> tgt = Tensor<>({  2,  4,  6,  8,  0 }).resize(shape);
	Tensor<> sqd = Tensor<>({  1,  4,  9, 16, 25 }).resize(shape);
	Tensor<> dif = Tensor<>({ -2, -4, -6, -8, 10 }).resize(shape);
	MSE<> critic(shape, false);
	
	double mse = critic.forward(inp, tgt);
	NNHardAssert(mse == sqd.sum(), "MSE<>::forward with no average failed!");
	
	critic.average(true);
	mse = critic.forward(inp, tgt);
	NNHardAssert(mse == sqd.mean(), "MSE<>::forward with average failed!");
	
	critic.inputs({ 10, 10 });
	mse = critic.safeForward(inp, tgt);
	NNHardAssert(mse == sqd.mean(), "MSE<>::safeForward failed!");
	
	critic.average(false);
	critic.backward(inp, tgt);
	NNHardAssert(fabs(critic.inGrad().addMM(dif, -1).sum()) < 1e-12, "MSE<>::backward failed!");
	
	critic.inputs({ 10, 10 });
	critic.safeBackward(inp, tgt);
	NNHardAssert(fabs(critic.inGrad().addMM(dif, -1).sum()) < 1e-12, "MSE<>::safeBackward failed!");
	
	critic.batch(12);
	NNHardAssert(critic.batch() == 12, "MSE<>::batch failed!");
}
