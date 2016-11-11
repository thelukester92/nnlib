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
	
	result = bias + weights * input;
	
	cout << "Passed all tests!" << endl;
	
	return 0;
}
