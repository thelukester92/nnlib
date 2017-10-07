#ifndef TEST_NLL
#define TEST_NLL

#include "nnlib/critics/nll.hpp"
using namespace nnlib;

void TestNLL()
{
	Tensor<> inp = Tensor<>({ 1, 2, 3, 4, 5, 6 }).resize(2, 3);
	Tensor<> tgt = Tensor<>({ 2, 0 }).resize(2, 1);
	Tensor<> dif = Tensor<>({ -3, -4 }).resize(2, 1);
	Tensor<> grd = Tensor<>({ 0, 0, -1.0, -1.0, 0, 0 }).resize(2, 3);
	NLL<> critic(false);
	
	double nll = critic.forward(inp, tgt);
	NNHardAssert(fabs(nll - dif.sum()) < 1e-12, "NLL<>::forward with no average failed!");
	
	critic.average(true);
	nll = critic.forward(inp, tgt);
	dif.scale(1.0/6.0);
	NNHardAssert(fabs(nll - dif.sum()) < 1e-12, "NLL<>::forward with average failed!");
	
	critic.average(false);
	critic.backward(inp, tgt);
	NNHardAssert(critic.inGrad().addM(grd, -1).square().sum() < 1e-12, "NLL<>::backward with no average failed!");
	
	critic.average(true);
	critic.backward(inp, tgt);
	NNHardAssert(critic.inGrad().addM(grd, -1.0 / inp.size()).square().sum() < 1e-12, "NLL<>::backward with average failed!");
}

#endif
