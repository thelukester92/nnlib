#ifndef TEST_NADAM_H
#define TEST_NADAM_H

#include "nnlib/opt/nadam.h"
#include "nnlib/nn/linear.h"
#include "nnlib/critics/mse.h"
using namespace nnlib;

void TestNadam()
{
	RandomEngine::seed();
	
	Tensor<> feat = Tensor<>(10, 2).rand();
	Tensor<> lab = Tensor<>(10, 3).rand();
	
	Linear<> nn(2, 3);
	MSE<> critic;
	
	double errBefore = critic.forward(nn.forward(feat), lab);
	
	Nadam<> opt(nn, critic);
	opt.step(feat.narrow(0, 0), lab.narrow(0, 0));
	opt.step(feat, lab);
	
	double errAfter = critic.forward(nn.forward(feat), lab);
	
	NNAssertLessThan(errAfter - errBefore, 0, "Optimization failed!");
}

#endif
