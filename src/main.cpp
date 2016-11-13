#include "tensor.h"
#include "error.h"
#include "random.h"
#include "loader.h"
#include "linear.h"
#include "tanh.h"
#include "sequential.h"
#include "squared_error.h"
#include <iostream>
#include <chrono>
using namespace nnlib;
using namespace std;

void testCorrectness();
double testEfficiency(size_t inps, size_t outs, size_t epochs, function<void()> &start, function<void()> &end);
void testMNIST();

int main()
{
	size_t inps				= 10000;
	size_t outs				= 1000;
	size_t epochs			= 100;
	
	using clock = chrono::high_resolution_clock;
	clock::time_point start;
	function<void()> startFn = [&](void) { start = clock::now(); };
	function<void()> endFn = [&](void) { cout << "took " << chrono::duration<double>(clock::now() - start).count() / epochs << " seconds per epoch" << endl; };
	
	testCorrectness();
	testEfficiency(inps, outs, epochs, startFn, endFn);
	testMNIST();
	
	return 0;
}

void testCorrectness()
{
	size_t inps = 2, outs = 3;
	Linear<double> layer(inps, outs);
	Vector<double> input(inps), target(outs);
	
	Vector<double> allWeights = Vector<double>::flatten(layer.parameters());
	size_t i = 0;
	
	// weights
	allWeights(i++) = 1;
	allWeights(i++) = 0;
	allWeights(i++) = 0;
	allWeights(i++) = 1;
	allWeights(i++) = 1;
	allWeights(i++) = 1;
	
	// bias
	allWeights(i++) = 0;
	allWeights(i++) = 1;
	allWeights(i++) = 2;
	
	input(0) = 3.14;
	input(1) = 10.0;
	
	target(0) = 3.14;
	target(1) = 11.0;
	target(2) = 15.14;
	
	Vector<double> &result = layer.forward(input);
	for(size_t i = 0; i < outs; ++i)
		Assert(result(i) == target(i), "forward failed!");
	
	Vector<double> &blame = layer.backward(input, target);
	Assert(blame(0) == 18.28 && blame(1) == 26.14, "backward failed to assign correct input blame!");
	
	Vector<double> &biasBlame = *(Vector<double> *)layer.blame()[1];
	for(size_t i = 0; i < outs; ++i)
		Assert(biasBlame(i) == target(i), "backward failed to assign correct bias blame!");
	
	Matrix<double> expectedBlame(outs, inps);
	expectedBlame[0] = 9.8596;
	expectedBlame[1] = 31.4;
	expectedBlame[2] = 34.54;
	expectedBlame[3] = 110.0;
	expectedBlame[4] = 47.5396;
	expectedBlame[5] = 151.4;
	
	Matrix<double> &weightsBlame = *(Matrix<double> *)layer.blame()[0];
	for(size_t i = 0; i < expectedBlame.size(); ++i)
		Assert(fabs(weightsBlame[i] - expectedBlame[i]) < 1e-12, "backward failed to assign correct weights blame!");
	
	Tanh<double> activation(outs);
	Vector<double> &act = activation.forward(layer.output());
	for(size_t i = 0; i < act.size(); ++i)
		Assert(act[i] == tanh(layer.output()[i]), "tanh forward failed!");
	
	Vector<double> &tanhBlame = activation.backward(layer.output(), target);
	for(size_t i = 0; i < tanhBlame.size(); ++i)
		Assert(tanhBlame[i] == (1.0 - act[i] * act[i]), "tanh backward failed!");
	
	Sequential<double> nn;
	nn.add(&layer);
	nn.add(&activation);
	
	Vector<double> tanOut = activation.forward(layer.forward(input));
	Vector<double> seqOut = nn.forward(input);
	for(size_t i = 0; i < seqOut.size(); ++i)
		Assert(tanOut(i) == seqOut(i), "sequential forward failed!");
	
	Vector<double> layIn = layer.backward(input, activation.backward(layer.output(), target));
	Vector<double> seqIn = nn.backward(input, target);
	for(size_t i = 0; i < seqIn.size(); ++i)
		Assert(layIn(i) == seqIn(i), "sequential backward failed!");
	
	// release the layers, since they weren't dynamically allocated!
	nn.release(1);
	nn.release(0);
	
	cout << "Passed all tests!" << endl;
}

double testEfficiency(size_t inps, size_t outs, size_t epochs, function<void()> &start, function<void()> &end)
{
	Matrix<double> weights(outs, inps);
	Vector<double> input(inps), bias(outs), result(outs);
	Random r;
	
	r.fillNormal(weights);
	r.fillNormal(bias);
	r.fillNormal(input);
	result.fill(0.0);
	
	start();
	for(size_t i = 0; i < epochs; ++i)
		result += weights * input + bias;
	end();
	
	double resultSum = 0.0;
	for(size_t i = 0; i < outs; ++i)
		resultSum += result[i];
	
	return resultSum;
}

void testMNIST()
{
	Sequential<double> nn;
	nn.add(
		new Linear<double>(784, 300), new Tanh<double>(300),
		new Linear<double>(300, 100), new Tanh<double>(100),
		new Linear<double>(100, 10), new Tanh<double>(10)
	);
	SquaredError<double> critic(10);
	
	Matrix<double> train = Loader<double>::loadRaw("../datasets/mnist/train.raw");
	Matrix<double> test = Loader<double>::loadRaw("../datasets/mnist/test.raw");
	
	Matrix<double> trainFeat(train.rows(), train.cols() - 1), trainLab(train.rows(), 10);
	Matrix<double> testFeat(test.rows(), test.cols() - 1), testLab(test.rows(), 10);
	
	trainLab.fill(0);
	for(size_t i = 0; i < train.rows(); ++i)
	{
		for(size_t j = 0; j < trainFeat.cols(); ++j)
			trainFeat(i, j) = train(i, j);
		trainLab(i, train(i, trainFeat.cols())) = 1;
	}
	
	testLab.fill(0);
	for(size_t i = 0; i < test.rows(); ++i)
	{
		for(size_t j = 0; j < testFeat.cols(); ++j)
			testFeat(i, j) = test(i, j);
		testLab(i, test(i, testFeat.cols())) = 1;
	}
	
	Vector<double> parameters = Vector<double>::flatten(nn.parameters());
	Vector<double> blame = Vector<double>::flatten(nn.blame());
	
	RandomIterator ri(train.rows());
	Vector<double> feat, lab;
	size_t n = 0;
	
	for(auto i : ri)
	{
		trainFeat.row(i, feat);
		trainLab.row(i, lab);
		
		nn.forward(feat);
		nn.backward(feat, critic.backward(nn.output(), lab));
		
		auto p = parameters.begin();
		auto b = blame.begin();
		
		for(; p != parameters.end(); ++p, ++b)
			*p -= 0.01 * *b;
		
		if(++n > 1000)
			break;
	}
}
