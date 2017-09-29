#ifndef TEST_CRITIC_SEQUENCER_H
#define TEST_CRITIC_SEQUENCER_H

#include "nnlib/critics/criticsequencer.h"
#include "nnlib/critics/mse.h"
using namespace nnlib;

void TestCriticSequencer()
{
	Storage<size_t> shape = { 5, 1, 1 };
	Tensor<> inp = Tensor<>({  1,  2,  3,  4,  5 }).resize(shape);
	Tensor<> tgt = Tensor<>({  2,  4,  6,  8,  0 }).resize(shape);
	Tensor<> sqd = Tensor<>({  1,  4,  9, 16, 25 }).resize(shape);
	Tensor<> dif = Tensor<>({ -2, -4, -6, -8, 10 }).resize(shape);
	MSE<> *innerCritic = new MSE<>(false);
	CriticSequencer<> critic(innerCritic);
	
	double mse = critic.forward(inp, tgt);
	NNAssertAlmostEquals(mse, sqd.sum(), 1e-12, "CriticSequencer::forward with no average failed!");
	
	critic.backward(inp, tgt);
	NNAssert(critic.inGrad().reshape(5, 1).addM(dif.reshape(5, 1), -1).square().sum() < 1e-12, "CriticSequencer::backward failed!");
}

#endif
