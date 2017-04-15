#include <iostream>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

void testTensor()
{
	Tensor<double> tensor(5, 3, 2);
	NNAssert(tensor.size() == 5*3*2, "Tensor::Tensor(...) yielded the wrong tensor size!");
	NNAssert(tensor.size(0) == 5, "Tensor::Tensor(...) yielded the wrong 0th dimension size!");
	NNAssert(tensor.size(1) == 3, "Tensor::Tensor(...) yielded the wrong 1st dimension size!");
	NNAssert(tensor.size(2) == 2, "Tensor::Tensor(...) yielded the wrong 2nd dimension size!");
	
	
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
