#ifndef TEST_ADAM_H
#define TEST_ADAM_H

#include "nnlib/opt/adam.h"
#include "nnlib/nn/linear.h"
#include "nnlib/critics/mse.h"
using namespace nnlib;

void TestAdam()
{
	RandomEngine::seed();
	
	Tensor<> feat = Tensor<>(10, 2).rand();
	Tensor<> lab = Tensor<>(10, 3).rand();
	
	Linear<> nn(2, 3);
	MSE<> critic(nn.outputs());
	
	double errBefore = critic.safeForward(nn.safeForward(feat), lab);
	
	Adam<> opt(nn, critic);
	opt.batch(1);
	opt.step(feat.narrow(0, 0), lab.narrow(0, 0));
	opt.safeStep(feat, lab);
	
	double errAfter = critic.safeForward(nn.safeForward(feat), lab);
	
	NNAssertLessThan(errAfter - errBefore, 0, "Optimization failed!");
}

#endif
