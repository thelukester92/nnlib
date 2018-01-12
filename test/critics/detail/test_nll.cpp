#include "../test_nll.hpp"
#include "nnlib/critics/nll.hpp"
using namespace nnlib;

void TestNLL()
{
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ 1, 2, 3, 4, 5, 6 }).resize(2, 3);
	Tensor<NN_REAL_T> tgt = Tensor<NN_REAL_T>({ 2, 0 }).resize(2, 1);
	Tensor<NN_REAL_T> dif = Tensor<NN_REAL_T>({ -3, -4 }).resize(2, 1);
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 0, 0, -1.0, -1.0, 0, 0 }).resize(2, 3);
	NLL<NN_REAL_T> critic(false);

	double nll = critic.forward(inp, tgt);
	NNHardAssert(fabs(nll - dif.sum()) < 1e-12, "NLL<NN_REAL_T>::forward with no average failed!");

	critic.average(true);
	nll = critic.forward(inp, tgt);
	dif.scale(1.0/6.0);
	NNHardAssert(fabs(nll - dif.sum()) < 1e-12, "NLL<NN_REAL_T>::forward with average failed!");

	critic.average(false);
	critic.backward(inp, tgt);
	NNHardAssert((critic.inGrad() - grd).square().sum() < 1e-12, "NLL<NN_REAL_T>::backward with no average failed!");

	critic.average(true);
	critic.backward(inp, tgt);
	NNHardAssert((critic.inGrad() - grd / inp.size()).square().sum() < 1e-12, "NLL<NN_REAL_T>::backward with average failed!");
}
