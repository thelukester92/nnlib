#include "tensor.h"
#include "error.h"
#include "random.h"
#include <iostream>
#include <chrono>
using namespace nnlib;
using namespace std;

void testCorrectness();
double testEfficiency(size_t inps, size_t outs, size_t epochs, function<void()> &start, function<void()> &end);

int main()
{
	size_t inps				= 100000;
	size_t outs				= 1000;
	size_t epochs			= 10;
	
	using clock = chrono::high_resolution_clock;
	clock::time_point start;
	function<void()> startFn = [&](void) { start = clock::now(); };
	function<void()> endFn = [&](void) { cout << "took " << chrono::duration<double>(clock::now() - start).count() / epochs << " seconds per epoch" << endl; };
	
	testCorrectness();
	testEfficiency(inps, outs, epochs, startFn, endFn);
	
	return 0;
}

void testCorrectness()
{
	size_t inps = 2, outs = 3;
	Matrix<double> weights(outs, inps);
	Vector<double> input(inps), bias(outs), target(outs);
	
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
	
	Vector<double> temp(outs), temp2(outs);
	for(size_t i = 0; i < temp.size(); ++i)
		temp(i) = bias(i) + target(i);
	temp2 = bias + target;
	for(size_t i = 0; i < temp.size(); ++i)
		Assert(temp(i) == temp2(i), "Vector addition failed!");
	
	Vector<size_t> a(5), b(5);
	a = b;
	
	Vector<double> result = weights * input + bias;
	for(size_t i = 0; i < outs; ++i)
		Assert(result(i) == target(i), "forward failed!");
	
	cout << "Passed all tests!" << endl;
}

double testEfficiency(size_t inps, size_t outs, size_t epochs, function<void()> &start, function<void()> &end)
{
	/*
	Tensor<double> weights(outs, inps);
	Tensor<double> input(inps), bias(outs), result(outs);
	Random r;
	
	weights.fillNormal(r);
	input.fillNormal(r);
	bias.fillNormal(r);
	result.fill(0.0);
	
	start();
	for(size_t i = 0; i < epochs; ++i)
		result += weights * input + bias;
	end();
	
	double resultSum = 0.0;
	for(size_t i = 0; i < outs; ++i)
		resultSum += result[i];
	
	return resultSum;
	*/
	
	return 0.0;
}
