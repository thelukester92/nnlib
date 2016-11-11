#include "tensor.h"
#include "error.h"
#include <iostream>
using namespace nnlib;
using namespace std;

int main()
{
	Tensor<double> weights(10, 5), input(5), bias(10), result(10);
	weights.fillNormal(0.0, 1.0, 3.0);
	input.fillNormal(0.0, 1.0, 3.0);
	bias.fillNormal(0.0, 1.0, 3.0);
	result.fill(0.0);
	
	result += weights * input + bias;
	result += weights * input + weights * input;
	result += bias + bias;
	result += (bias + bias) + (weights * input + (bias + result));
	
	cout << "Passed all tests!" << endl;
	
	return 0;
}
