#include <iostream>
#include <vector>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

/// \todo RMSProp

int main()
{
	cout << "========== Sanity Test ==========" << endl;
	
	size_t inps = 3;
	size_t outs = 2;
	size_t batch = 5;
	
	Linear<double> layer1(inps, outs, batch);
	
	Vector<double> &bias = *(Vector<double> *)layer1.parameters()[0];
	Matrix<double> &weights = *(Matrix<double> *)layer1.parameters()[1];
	
	Vector<double> parameters(layer1.parameters());
	for(double &val : parameters)
		val = (rand() % 1000) / 500.0 - 1;
	
	for(double &val : bias)
		val = (rand() % 1000) / 500.0 - 1;
	for(double &val : weights)
		val = (rand() % 1000) / 500.0 - 1;
	
	Matrix<double> inputs(batch, inps);
	for(double &val : inputs)
		val = (rand() % 1000) / 500.0 - 1;
	
	Matrix<double> blame(batch, outs);
	for(double &val : blame)
		val = (rand() % 1000) / 500.0 - 1;
	
	Matrix<double> targets(batch, outs);
	targets.fill(-0.25);
	
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
	
	TanH<double> layer2(outs, batch);
	Sequential<double> nn;
	nn.add(new Linear<double>(layer1));
	nn.add(new TanH<double>(layer2));
	
	SSE<double> critic(outs, batch);
	SGD<Module<double>, SSE<double>> optimizer(nn, critic);
	
	nn.forward(inputs);
	for(size_t i = 0; i < batch; ++i)
		for(size_t j = 0; j < outs; ++j)
			NNAssert(fabs(nn.output()(i, j) - tanh(layer1.output()(i, j))) < 1e-6, "Sequential::forward failed!");
	cout << "Sequential::forward passed!" << endl;
	
	for(size_t i = 0; i < 10000; ++i)
	{
		Matrix<double>::shuffleRows(inputs, targets);
		optimizer.optimize(inputs, targets);
	}
	NNAssert(critic.forward(nn.forward(inputs), targets).sum() < 1.25, "SGD::optimize failed!");
	cout << "SGD::optimize passed!" << endl;
	
	cout << "Sanity test passed!" << endl << endl;
	
	// MARK: MNIST Test
	
	{
		cout << "========== MNIST Test ==========" << endl;
		cout << "Loading data..." << flush;
		
		Matrix<double> train = Loader<double>::loadArff("../datasets/mnist/train.arff");
		Matrix<double> test  = Loader<double>::loadArff("../datasets/mnist/test.arff");
		
		Matrix<double> trainFeat(train.rows(), train.cols() - 1), trainLab(train.rows(), 10, 0.0);
		Matrix<double> testFeat(test.rows(), test.cols() - 1), testLab(test.rows(), 10, 0.0);
		
		cout << " Done!\nPreprocessing data..." << flush;
		
		for(size_t i = 0; i < train.rows(); ++i)
		{
			size_t j;
			for(j = 0; j < trainFeat.cols(); ++j)
				trainFeat(i, j) = train(i, j) / 255.0;
			trainLab[train(i, j)] = 1.0;
		}
		
		for(size_t i = 0; i < test.rows(); ++i)
		{
			size_t j;
			for(j = 0; j < testFeat.cols(); ++j)
				testFeat(i, j) = test(i, j) / 255.0;
			testLab[test(i, j)] = 1.0;
		}
		
		cout << " Done!\nCreating network..." << flush;
		
		Sequential<> nn;
		nn.add(
			new Linear<>(trainFeat.cols(), 300), new TanH<>(),
			new Linear<>(100), new TanH<>(),
			new Linear<>(10), new TanH<>()
		);
		
		SSE<double> critic(10);
		SGD<Module<double>, SSE<double>> optimizer(nn, critic);
		
		cout << " Done!\nInitial SSE: " << flush;
		nn.batch(testFeat.rows());
		critic.batch(testFeat.rows());
		cout << critic.forward(nn.forward(testFeat), testLab).sum() << endl;
		nn.batch(1);
		critic.batch(1);
		
		size_t epochs = 100;
		size_t presentationsPerEpoch = 200;
		
		cout << "Training..." << endl;
		for(size_t i = 0; i < epochs; ++i)
		{
			Matrix<double>::shuffleRows(trainFeat, trainLab);
			for(size_t j = 0; j < presentationsPerEpoch; ++j)
				optimizer.optimize(trainFeat[j], trainLab[j]);
			
			Progress::display(i, epochs);
			
			nn.batch(testFeat.rows());
			critic.batch(testFeat.rows());
			cout << "\t" << critic.forward(nn.forward(testFeat), testLab).sum() << flush;
			nn.batch(1);
			critic.batch(1);
		}
		Progress::display(epochs, epochs, '\n');
	}
	
	return 0;
}
