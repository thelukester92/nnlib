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
		cout << "========== Simple Concat ==========" << endl;
		
		Sequential<> *one = new Sequential<>(new Linear<>(1, 1));
		Sequential<> *two = new Sequential<>(new Linear<>(1, 1), new Sin<>());
		Concat<> *concat  = new Concat<>(one, two);
		Sequential<> nn(concat);
		
		dynamic_cast<Linear<> *>(one->component(0))->bias().fill(0);
		dynamic_cast<Linear<> *>(one->component(0))->weights().fill(1);
		
		dynamic_cast<Linear<> *>(two->component(0))->bias().fill(0);
		dynamic_cast<Linear<> *>(two->component(0))->weights().fill(1);
		
		Matrix<> input(1, 1);
		input(0, 0) = 1;
		
		Matrix<> blame(1, 2);
		blame(0, 0) = 1;
		blame(0, 1) = 0.5;
		
		nn.forward(input);
		nn.backward(input, blame);
		
		NNHardAssert(nn.output()(0, 0) == 1, "Linear forward in concat failed!");
		NNHardAssert(nn.output()(0, 1) == sin(1), "Sinusoid forward in concat failed!");
		NNHardAssert(nn.inputBlame()(0, 0) == 1 + 0.5 * cos(1), "Backward in concat failed!");
		
		cout << "Simple concat test passed!" << endl << endl;
	}
	
	// ND
	if(false)
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
		
		double biggest = trainLab(0, 0), smallest = trainLab(0, 0);
		for(auto d : trainLab)
			biggest = std::max(biggest, d), smallest = std::min(smallest, d);
		for(auto &d : trainLab)
			d = 10 * (d - smallest) / (biggest - smallest);
		for(auto &d : testLab)
			d = 10 * (d - smallest) / (biggest - smallest);
		
		Saver<>::saveArff(trainLab, "newtrain.arff");
		
		cout << " Done." << endl;
		
		cout << "Creating network..." << flush;
		Linear<> *sine = new Linear<>(1, train.rows()), *line = new Linear<>(1, 10), *out = new Linear<>(1);
		Concat<> *concat = new Concat<>(
			new Sequential<>(sine, new Sin<>()),
			new Sequential<>(line)
		);
		Sequential<> nn(concat, out);
		SSE<double> critic(1);
		auto optimizer = MakeOptimizer<SGD>(nn, critic);
		optimizer.learningRate(0.001);
		cout << " Done." << endl;
		
		cout << "Initializing weights..." << flush;
		{
			auto &bias = sine->bias();
			auto &weights = sine->weights();
			for(size_t i = 0; i < bias.size() / 2; ++i)
			{
				for(size_t j = 0; j < weights.cols(); ++j)
				{
					weights(2 * i, j)		= 2.0 * M_PI * (i + 1);
					weights(2 * i + 1, j)	= 2.0 * M_PI * (i + 1);
				}
				bias(2 * i)		= 0.5 * M_PI;
				bias(2 * i + 1)	= M_PI;
			}
		}
		line->bias().fill(0);
		line->weights().fill(1);
		out->bias().scale(0.001);
		out->weights().scale(0.001);
		cout << " Done." << endl;
		
		cout << "Initial SSE: " << flush;
		nn.batch(testFeat.rows());
		critic.batch(testFeat.rows());
		cout << critic.forward(nn.forward(testFeat), testLab).sum() << endl;
		
		size_t epochs = 1000;
		size_t batchesPerEpoch = train.rows();
		size_t batchSize = 1;
		
		double l1 = 0.01 * optimizer.learningRate();
		
		Batcher<double> batcher(trainFeat, trainLab, batchSize);
		nn.batch(batchSize);
		critic.batch(batchSize);
		
		cout << "Training..." << endl;
		
		for(size_t i = 0; i < epochs; ++i)
		{
			for(size_t j = 0; j < batchesPerEpoch; ++j)
			{
				for(auto &w : out->bias())
					w = w > 0 ? std::max(0.0, w - l1) : std::min(0.0, w + l1);
				for(auto &w : out->weights())
					w = w > 0 ? std::max(0.0, w - l1) : std::min(0.0, w + l1);
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
		
		nn.batch(testFeat.rows());
		nn.forward(testFeat);
		for(auto &d : nn.output())
			d = d * (biggest - smallest) / 10.0 + smallest;
		Saver<>::saveArff(nn.output(), "prediction.arff");
		
		cout << endl;
	}
	
	// Convolution
	{
		cout << "========== Convolution Test ==========" << endl;
		
		Matrix<double> features(1, 5*5*3);
		features(0) = {
			// red
			0, 1, 1, 0, 0,
			1, 0, 1, 0, 2,
			2, 1, 0, 2, 2,
			0, 2, 0, 2, 2,
			2, 2, 2, 2, 0,
			
			// green
			0, 2, 2, 0, 0,
			0, 0, 1, 2, 2,
			2, 0, 1, 1, 0,
			1, 1, 0, 1, 0,
			1, 1, 2, 2, 1,
			
			// blue
			1, 1, 2, 1, 1,
			0, 1, 0, 0, 2,
			0, 1, 1, 2, 2,
			2, 0, 1, 1, 2,
			1, 1, 1, 0, 0
		};
		
		Matrix<double> labels(1, 3*3*2);
		labels(0) = {
			0, 3, 3,
			0, 3, 16,
			1, 0, 5,
			
			1, -2, 4,
			-1, 0, 3,
			7, 4, -7
		};
		
		Convolution<double> conv(5, 5, 3, 3, 3, 2, 1, 1, 1);
		conv.kernels()(0) = {
			// red
			-1, 1, 1,
			1, 0, -1,
			0, 0, 0,
			
			// green
			1, 1, -1,
			0, 0, -1,
			0, -1, -1,
			
			// blue
			1, 0, -1,
			1, 1, 0,
			1, 1, 1,
			
			// bias
			1
		};
		
		conv.kernels()(1) = {
			// red
			-1, -1, 1,
			0, 1, 1,
			1, 0, -1,
			
			// green
			0, 1, 0,
			-1, 0, 1,
			1, 1, 0,
			
			// blue
			-1, 0, -1,
			0, 0, -1,
			-1, 0, -1,
			
			// bias
			0
		};
		
		cout << "features.sum() = " << features.sum() << endl;
		cout << "diff is " << SSE<double>(1).forward(conv.forward(features), labels).sum() << endl;
	}
	
	// MNIST
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
