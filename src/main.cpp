#include <iostream>
#include <math.h>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

void testTensor()
{
	// MARK: Basics
	
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
	
	// MARK: Narrowing
	
	Tensor<double> base(3, 5);
	double value = 0;
	for(double &val : base)
	{
		val = value++;
	}
	
	Tensor<double> narrowed1 = base.narrow(1, 1, 2);
	for(double &val : narrowed1)
	{
		cout << val << endl;
	}
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
	// MARK: Linear Test
	
	Linear<double> perceptron(3, 2);
	Tensor<double> &weights = perceptron.weights();
	
	Tensor<double> inp = { 1.0, 2.0, 3.14 };
	inp.resize(1, 3);
	
	Tensor<double> target = {
		weights(0, 0) * inp(0, 0) + weights(1, 0) * inp(0, 1) + weights(2, 0) * inp(0, 2),
		weights(0, 1) * inp(0, 0) + weights(1, 1) * inp(0, 1) + weights(2, 1) * inp(0, 2)
	};
	target.resize(1, 2);
	
	Tensor<double> blame = { 1.5, -88.0 };
	blame.resize(1, 2);
	
	Tensor<double> inBlame = {
		weights(0, 0) * blame(0, 0) + weights(0, 1) * blame(0, 1),
		weights(1, 0) * blame(0, 0) + weights(1, 1) * blame(0, 1),
		weights(2, 0) * blame(0, 0) + weights(2, 1) * blame(0, 1)
	};
	inBlame.resize(1, 3);
	
	perceptron.forward(inp);
	for(size_t i = 0; i < target.size(); ++i)
	{
		NNHardAssert(fabs(target(0, i) - perceptron.output()(0, i)) < 1e-9, "Linear::forward failed!");
	}
	
	perceptron.backward(inp, blame);
	for(size_t i = 0; i < target.size(); ++i)
	{
		NNHardAssert(fabs(inBlame(0, i) - perceptron.inBlame()(0, i)) < 1e-9, "Linear::backward failed!");
	}
	
	// MARK: TanH Test
	
	TanH<double> tanh;
	tanh.resizeInput(perceptron.output().shape());
	
	tanh.forward(perceptron.output());
	for(size_t i = 0; i < tanh.output().size(1); ++i)
	{
		NNHardAssert(fabs(tanh.output()(0, i) - ::tanh(perceptron.output()(0, i))) < 1e-9, "TanH::forward failed!");
	}
	
	tanh.backward(perceptron.output(), blame);
	for(size_t i = 0; i < tanh.inBlame().size(1); ++i)
	{
		double dy = blame(0, i) * (1.0 - ::tanh(perceptron.output()(0, i)) * ::tanh(perceptron.output()(0, i)));
		NNHardAssert(fabs(tanh.inBlame()(0, i) - dy) < 1e-9, "TanH::backward failed!");
	}
	
	// MARK: Sequential Test
	
	for(size_t i = 0; i < target.size(1); ++i)
	{
		target(0, i) = ::tanh(target(0, i));
	}
	
	double dy1 = 1.0 - ::tanh(perceptron.output()(0, 0)) * ::tanh(perceptron.output()(0, 0));
	double dy2 = 1.0 - ::tanh(perceptron.output()(0, 1)) * ::tanh(perceptron.output()(0, 1));
	
	inBlame = {
		weights(0, 0) * blame(0, 0) * dy1 + weights(0, 1) * blame(0, 1) * dy2,
		weights(1, 0) * blame(0, 0) * dy1 + weights(1, 1) * blame(0, 1) * dy2,
		weights(2, 0) * blame(0, 0) * dy1 + weights(2, 1) * blame(0, 1) * dy2
	};
	inBlame.resize(1, 3);
	
	Sequential<double> nn(&perceptron, &tanh);
	nn.forward(inp);
	nn.backward(inp, blame);
	
	for(size_t i = 0; i < target.size(); ++i)
	{
		NNHardAssert(fabs(target(0, i) - nn.output()(0, i)) < 1e-9, "Sequential::forward failed!");
	}
	
	for(size_t i = 0; i < inBlame.size(); ++i)
	{
		NNHardAssert(fabs(inBlame(0, i) - nn.inBlame()(0, i)) < 1e-9, "Sequential::backward failed!");
	}
	
	nn.remove(0);
	nn.remove(0);
}

int main()
{
	cout << "===== Testing Tensor =====" << endl;
	testTensor();
	cout << "Tensor test passed!" << endl;
	
	cout << "===== Testing Algebra =====" << endl;
	testAlgebra();
	cout << "Algebra test passed!" << endl;
	
	cout << "===== Testing Neural Networks =====" << endl;
	testNeuralNet();
	cout << "Neural networks test passed!" << endl;
	
	return 0;
}
