#include <iostream>
#include <vector>
#include <chrono>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

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
		val = Random<double>::normal(0, 1, 1);
	
	Matrix<double> inputs(batch, inps);
	for(double &val : inputs)
		val = Random<double>::normal(0, 1, 1);
	
	Matrix<double> blame(batch, outs);
	for(double &val : blame)
		val = Random<double>::normal(0, 1, 1);
	
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
	
	using clock = chrono::high_resolution_clock;
	chrono::time_point<clock> start;
	
	// MARK: Concat Test
	
	{
		cout << "========== Concat Test ==========" << endl;
		
		cout << "Loading data..." << flush;
		Matrix<> train = Loader<>::loadArff("../datasets/mackey-glass/train.arff");
		Matrix<> test  = Loader<>::loadArff("../datasets/mackey-glass/test.arff");
		cout << " Done." << endl;
		
		cout << "Preprocessing data..." << flush;
		Matrix<> trainFeat	= train.block(0, 0, train.rows(), 1);
		Matrix<> trainLab	= train.block(0, 1, train.rows(), 1);
		Matrix<> testFeat	= test.block(0, 0, test.rows(), 1);
		Matrix<> testLab	= test.block(0, 1, test.rows(), 1);
		cout << " Done." << endl;
		
		cout << "Creating network..." << flush;
		Concat<> *concat = new Concat<>(
			new Sequential<>(new Linear<>(1, 100), new Sin<>()),
			new Sequential<>(new Linear<>(1, 10))
		);
		Sequential<> nn(concat, new Linear<>(1));
		SSE<double> critic(1);
		auto optimizer = MakeOptimizer<RMSProp>(nn, critic);
		optimizer.learningRate(1e-6);
		cout << " Done." << endl;
		
		cout << "Initial SSE: " << flush;
		nn.batch(testFeat.rows());
		critic.batch(testFeat.rows());
		cout << critic.forward(nn.forward(testFeat), testLab).sum() << endl;
		
		size_t epochs = 2000;
		size_t batchesPerEpoch = train.rows();
		size_t batchSize = 1;
		
		Batcher<double> batcher(trainFeat, trainLab, batchSize);
		nn.batch(batchSize);
		critic.batch(batchSize);
		
		cout << "Training..." << endl;
		
		for(size_t i = 0; i < epochs; ++i)
		{
			for(size_t j = 0; j < batchesPerEpoch; ++j)
			{
				optimizer.optimize(batcher.features(), batcher.labels());
				batcher.next(true);
			}
			
			Progress::display(i, epochs);
			
			nn.batch(testFeat.rows());
			critic.batch(testFeat.rows());
			cout << "\t" << critic.forward(nn.forward(testFeat), testLab).sum() << flush;
			nn.batch(batchSize);
			critic.batch(batchSize);
		}
		Progress::display(epochs, epochs, '\n');
		
		cout << endl;
	}
	
	// MARK: MNIST Test
	
	{
		cout << "========== MNIST Test ==========" << endl;
		
		cout << "Loading data..." << flush;
		start = clock::now();
		Matrix<double> train = Loader<double>::loadArff("../datasets/mnist/train.arff");
		Matrix<double> test  = Loader<double>::loadArff("../datasets/mnist/test.arff");
		cout << " Done in " << chrono::duration<double>(clock::now() - start).count() << endl;
		
		cout << "Preprocessing data..." << flush;
		start = clock::now();
		
		Matrix<double> trainLab(train.rows(), 10, 0.0);
		Matrix<double> trainFeat = train.block(0, 0, train.rows(), train.cols() - 1);
		trainFeat.scale(1.0 / 255.0);
		
		Matrix<double> testLab(test.rows(), 10, 0.0);
		Matrix<double> testFeat = test.block(0, 0, test.rows(), test.cols() - 1);
		testFeat.scale(1.0 / 255.0);
		
		for(size_t i = 0; i < train.rows(); ++i)
			trainLab(i, train(i).back()) = 1.0;
		
		for(size_t i = 0; i < test.rows(); ++i)
			testLab(i, test(i).back()) = 1.0;
		
		cout << " Done in " << chrono::duration<double>(clock::now() - start).count() << endl;
		
		cout << "Creating network..." << flush;
		start = clock::now();
		
		Sequential<> nn;
		nn.add(
			new Linear<>(trainFeat.cols(), 300), new TanH<>(),
			new Linear<>(100), new TanH<>(),
			new Linear<>(10), new TanH<>()
		);
		
		SSE<double> critic(10);
		auto optimizer = MakeOptimizer<RMSProp>(nn, critic);
		
		cout << " Done in " << chrono::duration<double>(clock::now() - start).count() << endl;
		
		cout << "Initial SSE: " << flush;
		nn.batch(testFeat.rows());
		critic.batch(testFeat.rows());
		cout << critic.forward(nn.forward(testFeat), testLab).sum() << endl;
		
		size_t epochs = 100;
		size_t batchesPerEpoch = 100;
		size_t batchSize = 10;
		
		Batcher<double> batcher(trainFeat, trainLab, batchSize);
		nn.batch(batchSize);
		critic.batch(batchSize);
		
		cout << "Training..." << endl;
		start = clock::now();
		
		for(size_t i = 0; i < epochs; ++i)
		{
			for(size_t j = 0; j < batchesPerEpoch; ++j)
			{
				optimizer.optimize(batcher.features(), batcher.labels());
				batcher.next(true);
			}
			
			Progress::display(i, epochs);
			
			nn.batch(testFeat.rows());
			critic.batch(testFeat.rows());
			cout << "\t" << critic.forward(nn.forward(testFeat), testLab).sum() << flush;
			nn.batch(batchSize);
			critic.batch(batchSize);
		}
		Progress::display(epochs, epochs, '\n');
		
		cout << " Done in " << chrono::duration<double>(clock::now() - start).count() << endl;
	}
	
	return 0;
}
