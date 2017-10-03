#ifndef TEST_SGD_H
#define TEST_SGD_H

#include "nnlib/opt/sgd.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/critics/mse.hpp"
using namespace nnlib;

void TestSGD()
{
	RandomEngine::seed();
	
	Tensor<> feat = Tensor<>(10, 2).rand();
	Tensor<> lab = Tensor<>(10, 3).rand();
	
	Linear<> nn(2, 3);
	MSE<> critic;
	
	double errBefore = critic.forward(nn.forward(feat), lab);
	
	SGD<> opt(nn, critic);
	opt.step(feat.narrow(0, 0), lab.narrow(0, 0));
	opt.step(feat, lab);
	
	double errAfter = critic.forward(nn.forward(feat), lab);
	
	NNAssertLessThan(errAfter - errBefore, 0, "Optimization failed!");
	
	errBefore = errAfter;
	opt.momentum(0.25);
	opt.step(feat, lab);
	errAfter = critic.forward(nn.forward(feat), lab);
	
	NNAssertLessThan(errAfter - errBefore, 0, "Optimization failed!");
}

#endif
