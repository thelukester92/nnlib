#include <iostream>
#include <vector>
#include "matrix.h"
#include "linear.h"
using namespace std;
using namespace nnlib;

int main()
{
	size_t inps = 3;
	size_t outs = 2;
	size_t batch = 5;
	
	Linear<double> layer1(inps, outs, batch);
	Matrix<double> &weights = *layer1.parameters()[0];
	
	for(double &val : weights)
		val = (rand() % 1000) / 500.0 - 1;
	
	Matrix<double> inputs(batch, inps);
	layer1.forward(inputs);
	
	
	
	return 0;
}
