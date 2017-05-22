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
	MSE<> *innerCritic = new MSE<>({ 1, 1 }, false);
	CriticSequencer<> critic(innerCritic, 5);
	
	double mse = critic.forward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.sum()) < 1e-12, "CriticSequencer<>::forward with no average failed!");
	
	critic.inputs({ 10, 10, 10 });
	mse = critic.safeForward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.sum()) < 1e-12, "CriticSequencer<>::safeForward failed!");
	
	critic.backward(inp, tgt);
	NNHardAssert(fabs(critic.inGrad().reshape(5, 1).addM(dif.reshape(5, 1), -1).sum()) < 1e-12, "CriticSequencer<>::backward failed!");
	
	critic.inputs({ 10, 10, 10 });
	critic.safeBackward(inp, tgt);
	NNHardAssert(fabs(critic.inGrad().reshape(5, 1).addM(dif.reshape(5, 1), -1).sum()) < 1e-12, "CriticSequencer<>::safeBackward failed!");
	
	critic.batch(12);
	NNHardAssert(critic.batch() == 12, "CriticSequencer<>::batch failed!");
	
	critic.sequenceLength(10);
	NNHardAssert(critic.sequenceLength() == 10, "CriticSequencer<>::sequenceLength failed!");
	
	critic.safeForward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.sum()) < 1e-12, "CriticSequencer<>::safeForward failed!");
}

#endif
