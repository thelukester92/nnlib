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
	
	/*
	dx^i  = dyi * gamma
	dvar  = sum(dx^i * (xi - mean) * -1/2.0 * pow(var + 1e-12, -1.5))
	dmean = sum(dx^i * -1/sqrt(var + 1e-12)) + dvar * sum(-2 * (xi - mean) / bats)
	dxi   = dx^i * 1/sqrt(var + 1e-12) + dvar * 2*(xi - mean) / bats + dmean / m
	dgam  = sum(dyi * x^i)
	dbet  = sum(dyi)
	*/
	
	/*
	dvar  = sum(dyi * (xi - mean) * -1/2.0 * pow(var + 1e-12, -1.5))
	      = 
	
	dmean = sum(dyi * -1/sqrt(var + 1e-12)) + dvar * sum(-2 * (xi - mean) / bats)
	dxi   = dyi * 1/sqrt(var + 1e-12) + dvar * 2*(xi - mean) / bats + dmean / m
	dgam  = sum(dyi * x^i)
	      = <14.96 2.89 0>
	dbet  = sum(dyi)
	      = <10 5 12>
	*/
	
	/*
	Tensor<> inGrad = Tensor<>({
		
	}).reshape(3, 3);
	*/
	
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
		"BatchNorm::backward failed!"
	);
}
