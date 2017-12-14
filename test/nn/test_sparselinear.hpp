#ifndef TEST_SPARSE_LINEAR_H
#define TEST_SPARSE_LINEAR_H

#include "nnlib/nn/sparselinear.hpp"
#include "test_module.hpp"
using namespace nnlib;

void TestSparseLinear()
{
	// Sparse linear layer with arbitrary parameters
	SparseLinear<NN_REAL_T> module(2, 3);
	module.weights().copy({ -3, -2, 2, 3, 4, 5 });
	module.bias().copy({ -5, 7, 8862.37 });
	
	// Linear layer with the same parameters for comparison
	Linear<NN_REAL_T> linear(2, 3);
	linear.weights().copy(module.weights());
	linear.bias().copy(module.bias());
	
	// Arbitrary sparse input (batch)
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({
		3, 2, 0.0,
		0, 0, 3.14,
		0, 1, -5.9,
		1, 1, 4.3,
		2, 0, -1.0
	}).resize(5, 3);
	
	// Dense representation of inp
	Tensor<NN_REAL_T> dense = inp.unsparsify();
	
	// Arbitrary output gradient (batch)
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 1, 2, 3, -4, -3, 2, 5, 1, 5 }).resize(3, 3);
	
	// Test forward and backward using the parameters above
	
	module.forward(inp);
	linear.forward(dense);
	
	module.backward(inp, grd);
	linear.backward(dense, grd);
	
	NNAssertLessThan(module.output().copy().addM(linear.output(), -1).square().sum(), 1e-9, "SparseLinear::forward failed; wrong output!");
	NNAssertLessThan(module.inGrad().copy().addM(linear.inGrad(), -1).square().sum(), 1e-9, "SparseLinear::backward failed; wrong input gradient!");
	NNAssertLessThan(module.grad().addV(linear.grad(), -1).square().sum(), 1e-9, "SparseLinear::backward failed; wrong parameter gradient!");
	
	inp = Tensor<NN_REAL_T>({ 2, 0.0, 0, 3.14 }).resize(2, 2);
	dense = inp.unsparsify();
	grd = { 1, 2, 3 };
	
	module.forward(inp);
	linear.forward(dense);
	
	module.backward(inp, grd);
	linear.backward(dense, grd);
	
	NNAssertLessThan(module.output().copy().add(linear.output(), -1).square().sum(), 1e-9, "SparseLinear::forward failed for a vector; wrong output!");
	NNAssertLessThan(module.inGrad().copy().add(linear.inGrad(), -1).square().sum(), 1e-9, "SparseLinear::backward failed for a vector; wrong input gradient!");
	
	SparseLinear<NN_REAL_T> unbiased(2, 3, false);
	unbiased.weights().copy(module.weights());
	
	unbiased.forward(inp);
	unbiased.backward(inp, grd);
	
	NNAssertLessThan(unbiased.output().copy().add(module.bias()).add(linear.output(), -1).square().sum(), 1e-9, "SparseLinear::forward failed without bias; wrong output!");
	NNAssertLessThan(unbiased.inGrad().copy().add(linear.inGrad(), -1).square().sum(), 1e-9, "SparseLinear::backward failed without bias; wrong input gradient!");
	
	bool ok = true;
	try
	{
		module.forward(Tensor<NN_REAL_T>(1, 1));
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "SparseLinear::forward accepted an invalid input shape!");
	
	ok = true;
	try
	{
		module.backward(Tensor<NN_REAL_T>(1, 1), Tensor<NN_REAL_T>(1, 1));
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "SparseLinear::backward accepted invalid input and outGrad shapes!");
	
	TestModule("SparseLinear", module, inp, false);
}

#endif
