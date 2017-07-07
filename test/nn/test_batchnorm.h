#ifndef TEST_BATCHNORM_H
#define TEST_BATCHNORM_H

#include "nnlib/nn/batchnorm.h"
#include "test_module.h"
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
		 0.03596,  0.00000,  0,
		-0.02489, -2.12132,  0,
		-0.01106,  2.12132,  0
	}).resize(3, 3);
	
	Tensor<> paramGrad = Tensor<>({
		14.9606, 2.82843, 0,
		10, 5, 12
	});
	
	BatchNorm<> bn(3, 3);
	bn.weights().ones();
	bn.bias().zeros();
	bn.momentum(1.0);
	
	bn.forward(inp);
	for(size_t i = 0; i < 3; ++i)
	{
		NNAssertLessThan(fabs(bn.output().select(1, i).mean()), 1e-9, "BatchNorm::forward failed! Non-zero mean!");
		NNAssertAlmostEquals(bn.output().select(1, i).variance(), 1, 1e-9, "BatchNorm::forward failed! Non-unit variance!");
	}
	
	bn.backward(inp, grad);
	NNAssertLessThan(bn.grad().add(paramGrad, -1).square().sum(), 1e-9, "BatchNorm::backward failed! Wrong parameter gradient!");
	NNAssertLessThan(bn.inGrad().add(inGrad, -1).square().sum(), 1e-9, "BatchNorm::backward failed! Wrong input gradient!");
	
	bn.training(false);
	
	paramGrad.copy({
		12.21528, 2.30940, 0,
		10, 5, 12
	});
	
	inGrad.copy({
		 0.30038, 5.19615, 1.10940,
		-0.30038, 0.00000, 1.10940,
		 1.50188, 3.46410, 1.10940
	});
	
	bn.forward(inp);
	for(size_t i = 0; i < 3; ++i)
	{
		NNAssertLessThan(fabs(bn.output().select(1, i).mean()), 1e-9, "BatchNorm::forward (inference) failed! Non-zero mean!");
		NNAssertAlmostEquals(
			bn.output().select(1, i).scale(sqrt(3) / sqrt(2)).variance(), 1, 1e-9,
			"BatchNorm::forward (inference) failed! Non-unit variance!"
		);
	}
	
	bn.backward(inp, grad);
	NNAssertLessThan(bn.grad().add(paramGrad, -1).square().sum(), 1e-9, "BatchNorm::backward (inference) failed! Wrong parameter gradient!");
	NNAssertLessThan(bn.inGrad().add(inGrad, -1).square().sum(), 1e-9, "BatchNorm::backward (inference) failed! Wrong input gradient!");
	
	Storage<size_t> dims = { 3, 6 };
	bn.resize(dims, dims);
	NNAssertEquals(bn.inputs(), dims, "BatchNorm::resize failed!");
	NNAssertEquals(bn.outputs(), dims, "BatchNorm::resize failed!");
	
	dims = { 4, 4 };
	bn.inputs(dims);
	NNAssertEquals(bn.inputs(), dims, "BatchNorm::inputs failed!");
	NNAssertEquals(bn.outputs(), dims, "BatchNorm::inputs failed!");
	
	dims = { 5, 7 };
	bn.outputs(dims);
	NNAssertEquals(bn.inputs(), dims, "BatchNorm::outputs failed!");
	NNAssertEquals(bn.outputs(), dims, "BatchNorm::outputs failed!");
	
	dims = { 100, 7 };
	bn.safeResize(dims, dims);
	NNAssertEquals(bn.inputs(), dims, "BatchNorm::safeResize failed!");
	NNAssertEquals(bn.outputs(), dims, "BatchNorm::safeResize failed!");
	
	Tensor<> state = bn.state();
	state.fill(0);
	NNAssertAlmostEquals(bn.output().sum(), 0, 1e-12, "BatchNorm::state failed!");
	
	TestSerializationOfModule(bn);
	TestModule(bn);
}

#endif
