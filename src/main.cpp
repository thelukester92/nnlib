#include <iostream>
#include <math.h>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

void testTensor()
{
	Tensor<double> vector(5);
	NNHardAssert(vector.size() == 5, "Tensor::Tensor yielded the wrong tensor size!");
	NNHardAssert(vector.dims() == 1, "Tensor::Tensor yielded the wrong number of dimensions!");
	NNHardAssert(vector.size(0) == 5, "Tensor::Tensor yielded the wrong 0th dimension size!");
	
	vector.fill(3.14);
	for(const double &value : vector)
	{
		NNHardAssert(fabs(value - 3.14) < 1e-9, "Tensor::fill failed!");
	}
	
	Tensor<double> tensor(6, 3, 2);
	NNHardAssert(tensor.size() == 6*3*2, "Tensor::Tensor yielded the wrong tensor size!");
	NNHardAssert(tensor.dims() == 3, "Tensor::Tensor yielded the wrong number of dimensions!");
	NNHardAssert(tensor.size(0) == 6, "Tensor::Tensor yielded the wrong 0th dimension size!");
	NNHardAssert(tensor.size(1) == 3, "Tensor::Tensor yielded the wrong 1st dimension size!");
	NNHardAssert(tensor.size(2) == 2, "Tensor::Tensor yielded the wrong 2nd dimension size!");
	
	Tensor<double> reshaped = tensor.reshape(9, 4);
	NNHardAssert(reshaped.dims() == 2, "Tensor::reshape yielded the wrong number of dimensions!");
	NNHardAssert(reshaped.size(0) == 9, "Tensor::reshape yielded the wrong 0th dimension size!");
	NNHardAssert(reshaped.size(1) == 4, "Tensor::reshape yielded the wrong 1st dimension size!");
	
	bool causedProblems = false;
	try
	{
		tensor.reshape(3, 3);
	}
	catch(const std::runtime_error &e)
	{
		causedProblems = true;
	}
	NNHardAssert(causedProblems, "Tensor::reshape failed to yield an error for an incompatible shape!");
}

template <bool TransA, bool TransB>
void slowMatrixMultiply(Tensor<double> &A, Tensor<double> &B, Tensor<double> &C)
{
	C.fill(0);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			for(size_t k = 0; k < (TransA ? A.size(0) : A.size(1)); ++k)
			{
				C(i, j) += (TransA ? A(k, i) : A(i, k)) * (TransB ? B(j, k) : B(k, j));
			}
		}
	}
}

void testAlgebra()
{
	Tensor<double> A(10, 5);
	Tensor<double> B(5, 3);
	Tensor<double> C(10, 3);
	Tensor<double> C2(10, 3);
	
	A.rand();
	B.rand();
	
	slowMatrixMultiply<false, false>(A, B, C);
	Algebra<double>::gemm(A, B, C2);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			NNHardAssert(fabs(C(i, j) - C2(i, j)) < 1e-9, "Algebra::gemm failed!");
		}
	}
	
	B.resize(3, 5);
	slowMatrixMultiply<false, true>(A, B, C);
	Algebra<double>::gemmNT(A, B, C2);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			NNHardAssert(fabs(C(i, j) - C2(i, j)) < 1e-9, "Algebra::gemmNT failed!");
		}
	}
	
	A.resize(5, 10);
	slowMatrixMultiply<true, true>(A, B, C);
	Algebra<double>::gemmTT(A, B, C2);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			NNHardAssert(fabs(C(i, j) - C2(i, j)) < 1e-9, "Algebra::gemmTT failed!");
		}
	}
	
	B.resize(5, 3);
	slowMatrixMultiply<true, false>(A, B, C);
	Algebra<double>::gemmTN(A, B, C2);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			NNHardAssert(fabs(C(i, j) - C2(i, j)) < 1e-9, "Algebra::gemmTN failed!");
		}
	}
}

void testNeuralNet()
{
	Linear<double> perceptron;
	
	/*
	Sequential neuralDecomposition(
		new Concat(
			new Sequential(
				new Linear(100),
				new Activation<Sin>()
			),
			new Sequential(
				new Linear(10),
				new Activation<Tanh>()
			),
			new Linear(10)
		),
		new Linear(10)
	);
	*/
}

int main()
{
	cout << "===== Testing Tensor =====" << endl;
	testTensor();
	cout << "Tensor test passed!" << endl;
	
	cout << "===== Testing Algebra =====" << endl;
	testAlgebra();
	cout << "Algebra test passed!" << endl;
	
	testNeuralNet();
	return 0;
}
