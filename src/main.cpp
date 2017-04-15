#include <iostream>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

void testTensor()
{
	Tensor<double> tensor(6, 3, 2);
	NNAssert(tensor.size() == 6*3*2, "Tensor::Tensor yielded the wrong tensor size!");
	NNAssert(tensor.dims() == 3, "Tensor::Tensor yielded the wrong number of dimensions!");
	NNAssert(tensor.size(0) == 6, "Tensor::Tensor yielded the wrong 0th dimension size!");
	NNAssert(tensor.size(1) == 3, "Tensor::Tensor yielded the wrong 1st dimension size!");
	NNAssert(tensor.size(2) == 2, "Tensor::Tensor yielded the wrong 2nd dimension size!");
	
	tensor.reshape(9, 4);
	NNAssert(tensor.dims() == 2, "Tensor::reshape yielded the wrong number of dimensions!");
	NNAssert(tensor.size(0) == 9, "Tensor::reshape yielded the wrong 0th dimension size!");
	NNAssert(tensor.size(1) == 4, "Tensor::reshape yielded the wrong 1st dimension size!");
	
	bool causedProblems = false;
	try
	{
		tensor.reshape(3, 3);
	}
	catch(const std::runtime_error &e)
	{
		causedProblems = true;
	}
	NNAssert(causedProblems, "Tensor::reshape failed to yield an error for an incompatible shape!");
}

void testNeuralNet()
{
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
	testTensor();
	testNeuralNet();
	return 0;
}
