#include "nnlib/nn/batchnorm.h"
using namespace nnlib;

void TestBatchNorm()
{
	Tensor<> inp = Tensor<>({
		 3,  6,  9,
		-1,  5,  4,
		12,  5, 11
	}).resize(3, 3);
	
	Tensor<> grad = Tensor<>({
		 2,  3,  4,
		-2,  0,  4,
		10,  2,  4
	}).resize(3, 3);
	
	Tensor<> inGrad = Tensor<>({
		 0.0360,  0.0001,  0,
		-0.0249, -2.1213,  0,
		-0.0111,  2.1212,  0
	}).resize(3, 3);
	
	BatchNorm<> bn(3, 3);
	
	bn.forward(inp);
	for(size_t i = 0; i < 3; ++i)
	{
		NNHardAssert(fabs(bn.output().select(1, i).mean()) < 1e-9, "BatchNorm::forward failed! Non-zero mean!");
		NNHardAssert(fabs(bn.output().select(1, i).variance() - 1) < 1e-9, "BatchNorm::forward failed! Non-unit variance!");
	}
	
	bn.backward(inp, grad);
	NNHardAssert(
		bn.grad().add(Tensor<>({ 14.9606, 2.82843, 0, 10, 5, 12 }), -1).square().sum() < 1e-9,
		"BatchNorm::backward failed! Wrong parameter gradient!"
	);
	NNHardAssert(
		bn.inGrad().add(inGrad, -1).square().sum() < 1e-9,
		"BatchNorm::backward failed! Wrong input gradient!"
	);
}
