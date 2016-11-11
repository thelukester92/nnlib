#include "tensor.h"
#include "error.h"
#include <iostream>
using namespace nnlib;
using namespace std;

int main()
{
	size_t inps = 2, outs = 3;
	Tensor<double> weights(outs, inps), input(inps), bias(outs), target(outs), result(outs);
	
	weights(0, 0) = 1;
	weights(0, 1) = 0;
	weights(1, 0) = 0;
	weights(1, 1) = 1;
	weights(2, 0) = 1;
	weights(2, 1) = 1;
	
	input(0) = 3.14;
	input(1) = 10.0;
	
	bias(0) = 0;
	bias(1) = 1;
	bias(2) = 2;
	
	target(0) = 3.14;
	target(1) = 11.0;
	target(2) = 15.14;
	
	result = weights * input + bias;
	
	for(size_t i = 0; i < outs; ++i)
		Assert(result(i) == target(i), "Linear::forward failed!");
	
	/*
	Tensor<double> weights(10, 5), input(5), bias(10), result(10);
	weights.fillNormal(0.0, 1.0, 3.0);
	input.fillNormal(0.0, 1.0, 3.0);
	bias.fillNormal(0.0, 1.0, 3.0);
	result.fill(0.0);
	
	result = weights * input + bias;
	
	cout << "Passed all tests!" << endl;
	*/
	
	return 0;
}
