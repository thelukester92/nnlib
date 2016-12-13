#include <iostream>
#include <vector>
#include "matrix.h"
#include "linear.h"
#include "sgd.h"
#include "sse.h"
using namespace std;
using namespace nnlib;

double calcSSE(const Matrix<double> &inputs, const Matrix<double> &targets, Module<double> &model)
{
	model.forward(inputs);
	double d = 0;
	auto i = targets.begin();
	auto j = model.output().begin(), end = model.output().end();
	for(; j != end; ++i, ++j)
		d += (*j - *i) * (*j - *i);
	return d;
}

int main()
{
	size_t inps = 3;
	size_t outs = 2;
	size_t batch = 5;
	
	Linear<double> layer1(inps, outs, batch);
	
	Vector<double> &bias = *(Vector<double> *)layer1.parameters()[0];
	Matrix<double> &weights = *(Matrix<double> *)layer1.parameters()[1];
	
	Vector<double> parameters(layer1.parameters());
	for(double &val : parameters)
		val = (rand() % 1000) / 500.0 - 1;
	
	Matrix<double> inputs(batch, inps);
	for(double &val : inputs)
		val = (rand() % 1000) / 500.0 - 1;
	
	Matrix<double> blame(batch, outs);
	for(double &val : blame)
		val = (rand() % 1000) / 500.0 - 1;
	
	Matrix<double> targets(batch, outs);
	for(double &val : targets)
		val = (rand() % 1000) / 500.0 - 1;
	
	Matrix<double> outputs(batch, outs);
	for(size_t i = 0; i < batch; ++i)
	{
		for(size_t j = 0; j < outs; ++j)
		{
			outputs(i, j) = bias(j);
			for(size_t k = 0; k < inps; ++k)
				outputs(i, j) += inputs(i, k) * weights(j, k);
		}
	}
	
	layer1.forward(inputs);
	for(size_t i = 0; i < batch; ++i)
		for(size_t j = 0; j < outs; ++j)
			NNAssert(fabs(outputs(i, j) - layer1.output()(i, j)) < 1e-6, "Linear::forward failed!");
	cout << "Linear::forward passed!" << endl;
	
	Matrix<double> inputBlame(batch, inps, 0);
	for(size_t i = 0; i < batch; ++i)
		for(size_t j = 0; j < inps; ++j)
			for(size_t k = 0; k < outs; ++k)
				inputBlame(i, j) += blame(i, k) * weights(k, j);
	
	layer1.backward(inputs, blame);
	for(size_t i = 0; i < batch; ++i)
		for(size_t j = 0; j < inps; ++j)
			NNAssert(fabs(inputBlame(i, j) - layer1.inputBlame()(i, j)) < 1e-6, "Linear::backword failed!");
	cout << "Linear::backward passed!" << endl;
	
	SSE<double> critic(batch, outs);
	SGD<Linear<double>, SSE<double>> optimizer(layer1, critic);
	
	double inputSum = 0, targetSum = 0;
	for(double val : inputs)
		inputSum += val;
	for(double val : targets)
		targetSum += val;
	
	for(size_t i = 0; i < 1000; ++i)
	{
		Matrix<double>::shuffleRows(inputs, targets);
		cout << i << "\t" << calcSSE(inputs, targets, layer1) << endl;
		optimizer.optimize(inputs, targets);
		
		double foo = 0;
		for(double val : inputs)
			foo += val;
		NNAssert(fabs(inputSum - foo) < 1e-6, "shuffleRows failed!");
		
		foo = 0;
		for(double val : targets)
			foo += val;
		NNAssert(fabs(targetSum - foo) < 1e-6, "shuffleRows failed!");
	}
	
	return 0;
}
