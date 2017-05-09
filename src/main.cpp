#include <iostream>
#include <math.h>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

/// This file is a series of unit tests for nnlib.
/// For better demos, see https://github.com/thelukester92/nnlib-examples.git.

void testTensor()
{
	// MARK: Basics
	
	Tensor<double> vector(5);
	NNHardAssert(vector.size() == 5, "Tensor::Tensor yielded the wrong tensor size!");
	NNHardAssert(vector.dims() == 1, "Tensor::Tensor yielded the wrong number of dimensions!");
	NNHardAssert(vector.size(0) == 5, "Tensor::Tensor yielded the wrong 0th dimension size!");
	
	vector.fill(3.14);
	for(const double &value : vector)
	{
		NNHardAssert(fabs(value - 3.14) < 1e-9, "Tensor::fill failed!");
	}
	
	Tensor<double> tensor(6, 3, 2);
	NNHardAssert(tensor.size() == 6*3*2, "Tensor::Tensor yielded the wrong tensor size!");
	NNHardAssert(tensor.dims() == 3, "Tensor::Tensor yielded the wrong number of dimensions!");
	NNHardAssert(tensor.size(0) == 6, "Tensor::Tensor yielded the wrong 0th dimension size!");
	NNHardAssert(tensor.size(1) == 3, "Tensor::Tensor yielded the wrong 1st dimension size!");
	NNHardAssert(tensor.size(2) == 2, "Tensor::Tensor yielded the wrong 2nd dimension size!");
	
	Tensor<double> reshaped = tensor.reshape(9, 4);
	NNHardAssert(reshaped.dims() == 2, "Tensor::reshape yielded the wrong number of dimensions!");
	NNHardAssert(reshaped.size(0) == 9, "Tensor::reshape yielded the wrong 0th dimension size!");
	NNHardAssert(reshaped.size(1) == 4, "Tensor::reshape yielded the wrong 1st dimension size!");
	
	// MARK: Narrowing
	
	Tensor<size_t> base(3, 5);
	size_t value = 0;
	for(size_t &val : base)
	{
		val = value++;
	}
	
	Tensor<size_t> narrowed1 = base.narrow(1, 1, 2);
	Tensor<size_t> expected1 = { 1, 2, 6, 7, 11, 12 };
	
	Tensor<size_t> narrowed2 = narrowed1.narrow(0, 1, 2);
	Tensor<size_t> expected2 = { 6, 7, 11, 12 };
	
	for(auto i = narrowed1.begin(), j = expected1.begin(), k = narrowed1.end(); i != k; ++i, ++j)
	{
		NNHardAssert(*i == *j, "Tensor::narrow failed!");
	}
	
	for(auto i = narrowed2.begin(), j = expected2.begin(), k = narrowed2.end(); i != k; ++i, ++j)
	{
		NNHardAssert(*i == *j, "Tensor::narrow failed!");
	}
	
	Tensor<size_t> subbed = base.sub({ { 1, 2 }, { 1, 2 } });
	for(auto i = subbed.begin(), j = expected2.begin(), k = subbed.end(); i != k; ++i, ++j)
	{
		NNHardAssert(*i == *j, "Tensor::sub failed!");
	}
	
	// MARK: Flattening
	
	Tensor<double> a = Tensor<double>(5).rand(), b = Tensor<double>(7, 3).rand();
	Tensor<double> aCopy = a.copy(), bCopy = b.copy();
	Tensor<double> c = Tensor<double>::flatten({ &a, &b });
	
	for(auto i = a.begin(), j = aCopy.begin(), end = a.end(); i != end; ++i, ++j)
	{
		NNHardAssert(fabs(*i - *j) < 1e-9, "Tensor::flatten failed!");
	}
	
	for(auto i = b.begin(), j = bCopy.begin(), end = b.end(); i != end; ++i, ++j)
	{
		NNHardAssert(fabs(*i - *j) < 1e-9, "Tensor::flatten failed!");
	}
	
	// MARK: Selection
	
	Tensor<double> orig = Tensor<double>(10, 5, 3).rand();
	Tensor<double> slice = orig.select(1, 3);
	
	NNHardAssert(slice.dims() == 2, "Tensor::select failed to set the correct number of dimensions!");
	NNHardAssert(slice.size(0) == 10, "Tensor::select failed to set the correct 0th dimension size!");
	NNHardAssert(slice.size(1) == 3, "Tensor::select failed to set the correct 1st dimension size!");
	
	for(size_t x = 0; x < 10; ++x)
	{
		for(size_t y = 0; y < 3; ++y)
		{
			NNHardAssert(fabs(orig(x, 3, y) - slice(x, y)) < 1e-9, "Tensor::select failed!");
		}
	}
	
	// MARK: Transposition
	
	Tensor<double> notTransposed = Tensor<double>(5, 3).rand();
	Tensor<double> transposed(3, 5);
	for(size_t i = 0; i < 3; ++i)
	{
		for(size_t j = 0; j < 5; ++j)
		{
			transposed(i, j) = notTransposed(j, i);
		}
	}
	
	Tensor<double> alsoTransposed = notTransposed.transpose();
	NNHardAssert(transposed.shape() == alsoTransposed.shape(), "Tensor::transpose failed!");
	for(size_t i = 0; i < 3; ++i)
	{
		for(size_t j = 0; j < 5; ++j)
		{
			NNHardAssert(fabs(transposed(i, j) - alsoTransposed(i, j)) < 1e-9, "Tensor::transpose failed!");
		}
	}
}

template <bool TransA, bool TransB>
void slowMatrixMultiply(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C)
{
	C.fill(0);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			for(size_t k = 0; k < (TransA ? A.size(0) : A.size(1)); ++k)
			{
				C(i, j) += (TransA ? A(k, i) : A(i, k)) * (TransB ? B(j, k) : B(k, j));
			}
		}
	}
}

void testAlgebra()
{
	Tensor<double> A(10, 5);
	Tensor<double> B(5, 3);
	Tensor<double> C(10, 3);
	Tensor<double> C2(10, 3);
	
	A.rand();
	B.rand();
	
	slowMatrixMultiply<false, false>(A, B, C);
	C2.multiplyMM(A, B);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			NNHardAssert(fabs(C(i, j) - C2(i, j)) < 1e-9, "Tensor::multiplyMM failed!");
		}
	}
	
	B.resize(3, 5);
	slowMatrixMultiply<false, true>(A, B, C);
	C2.multiplyMMT(A, B);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			NNHardAssert(fabs(C(i, j) - C2(i, j)) < 1e-9, "Tensor::multiplyMMT failed!");
		}
	}
	
	A.resize(5, 10);
	slowMatrixMultiply<true, true>(A, B, C);
	C2.multiplyMTMT(A, B);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			NNHardAssert(fabs(C(i, j) - C2(i, j)) < 1e-9, "Tensor::multiplyMTMT failed!");
		}
	}
	
	B.resize(5, 3);
	slowMatrixMultiply<true, false>(A, B, C);
	C2.multiplyMTM(A, B);
	
	for(size_t i = 0; i < C.size(0); ++i)
	{
		for(size_t j = 0; j < C.size(1); ++j)
		{
			NNHardAssert(fabs(C(i, j) - C2(i, j)) < 1e-9, "Tensor::multiplyMTM failed!");
		}
	}
	
	Tensor<double> vec1 = Tensor<double>(10).rand(), vec2 = Tensor<double>(5).rand();
	Tensor<double> outerProduct(10, 5);
	outerProduct.multiplyVTV(vec1, vec2);
	for(size_t i = 0; i < vec1.size(); ++i)
	{
		for(size_t j = 0; j < vec2.size(); ++j)
		{
			NNHardAssert(fabs(outerProduct(i, j) - vec1(i) * vec2(j)) < 1e-9, "Tensor::multiplyVTV failed!");
		}
	}
	
	Tensor<double> mv(outerProduct.size(0));
	Tensor<double> actual(outerProduct.size(0), 1);
	slowMatrixMultiply<false, false>(outerProduct, vec2.reshape(outerProduct.size(1), 1), actual);
	mv.multiplyMV(outerProduct, vec2);
	for(size_t i = 0; i < mv.size(); ++i)
	{
		NNHardAssert(fabs(mv(i) - actual(i, 0)) < 1e-9, "Tensor::multiplyMV failed!");
	}
	
	// Make sure tensor enforces contiguous matrices for multiplication
	
	Tensor<double> foo = Tensor<double>(10, 10, 10).rand();
	Tensor<double> bar = foo.select(2, 1);
	Tensor<double> bat = Tensor<double>(10, 10).rand();
	Tensor<double> baz(10, 10);
	Tensor<double> baaz(10, 10);
	
	bool problem = false;
	try
	{
		baz.multiplyMM(bar, bat);
	}
	catch(std::runtime_error &e)
	{
		problem = true;
	}
	NNAssert(problem, "Non-contiguous matrix multiplcation failed to raise an error!");
}

void testNeuralNet()
{
	// MARK: Linear Test
	
	Linear<double> perceptron(3, 2);
	Tensor<double> &weights = perceptron.weights();
	
	Tensor<double> inp = { 1.0, 2.0, 3.14 };
	inp.resize(1, 3);
	
	Tensor<double> target = {
		weights(0, 0) * inp(0, 0) + weights(1, 0) * inp(0, 1) + weights(2, 0) * inp(0, 2),
		weights(0, 1) * inp(0, 0) + weights(1, 1) * inp(0, 1) + weights(2, 1) * inp(0, 2)
	};
	target.resize(1, 2);
	
	Tensor<double> grad = { 1.5, -88.0 };
	grad.resize(1, 2);
	
	Tensor<double> inGrad = {
		weights(0, 0) * grad(0, 0) + weights(0, 1) * grad(0, 1),
		weights(1, 0) * grad(0, 0) + weights(1, 1) * grad(0, 1),
		weights(2, 0) * grad(0, 0) + weights(2, 1) * grad(0, 1)
	};
	inGrad.resize(1, 3);
	
	perceptron.forward(inp);
	for(size_t i = 0; i < target.size(); ++i)
	{
		NNHardAssert(fabs(target(0, i) - perceptron.output()(0, i)) < 1e-9, "Linear::forward failed!");
	}
	
	perceptron.backward(inp, grad);
	for(size_t i = 0; i < target.size(); ++i)
	{
		NNHardAssert(fabs(inGrad(0, i) - perceptron.inGrad()(0, i)) < 1e-9, "Linear::backward failed!");
	}
	
	// MARK: TanH Test
	
	TanH<double> tanh;
	tanh.inputs(perceptron.outputs());
	
	tanh.forward(perceptron.output());
	for(size_t i = 0; i < tanh.output().size(1); ++i)
	{
		NNHardAssert(fabs(tanh.output()(0, i) - ::tanh(perceptron.output()(0, i))) < 1e-9, "TanH::forward failed!");
	}
	
	tanh.backward(perceptron.output(), grad);
	for(size_t i = 0; i < tanh.inGrad().size(1); ++i)
	{
		double dy = grad(0, i) * (1.0 - ::tanh(perceptron.output()(0, i)) * ::tanh(perceptron.output()(0, i)));
		NNHardAssert(fabs(tanh.inGrad()(0, i) - dy) < 1e-9, "TanH::backward failed!");
	}
	
	// MARK: Sequential Test
	
	for(size_t i = 0; i < target.size(1); ++i)
	{
		target(0, i) = ::tanh(target(0, i));
	}
	
	double dy1 = 1.0 - ::tanh(perceptron.output()(0, 0)) * ::tanh(perceptron.output()(0, 0));
	double dy2 = 1.0 - ::tanh(perceptron.output()(0, 1)) * ::tanh(perceptron.output()(0, 1));
	
	inGrad = {
		weights(0, 0) * grad(0, 0) * dy1 + weights(0, 1) * grad(0, 1) * dy2,
		weights(1, 0) * grad(0, 0) * dy1 + weights(1, 1) * grad(0, 1) * dy2,
		weights(2, 0) * grad(0, 0) * dy1 + weights(2, 1) * grad(0, 1) * dy2
	};
	inGrad.resize(1, 3);
	
	Sequential<double> nn(&perceptron, &tanh);
	nn.forward(inp);
	nn.backward(inp, grad);
	
	for(size_t i = 0; i < target.size(); ++i)
	{
		NNHardAssert(fabs(target(0, i) - nn.output()(0, i)) < 1e-9, "Sequential::forward failed!");
	}
	
	for(size_t i = 0; i < inGrad.size(); ++i)
	{
		NNHardAssert(fabs(inGrad(0, i) - nn.inGrad()(0, i)) < 1e-9, "Sequential::backward failed!");
	}
	
	// avoid double deallocation since nn's modules were not dynamically allocated
	nn.remove(0);
	nn.remove(0);
	
	// MARK: Optimization Test
	
	RandomEngine::seed(0);
	
	Sequential<> trainNet(
		new Linear<>(5, 10), new TanH<>(),
		new Linear<>(3), new TanH<>(),
		new LogSoftMax<>()
	);
	
	Sequential<> targetNet(
		new Linear<>(5, 10), new TanH<>(),
		new Linear<>(3), new TanH<>(),
		new LogSoftMax<>()
	);
	
	MSE<> critic(trainNet.outputs());
	SGD<> optimizer(trainNet, critic);
	optimizer.learningRate(0.001);
	
	Tensor<double> testFeat = Tensor<double>(100, 5).rand();
	Tensor<double> testLab(100, 3);
	for(size_t i = 0; i < 100; ++i)
	{
		testLab.narrow(0, i).copy(targetNet.forward(testFeat.narrow(0, i)));
	}
	
	trainNet.batch(10);
	targetNet.batch(10);
	critic.batch(10);
	
	for(size_t i = 0; i < 1000; ++i)
	{
		Tensor<double> feat = Tensor<double>(10, 5).rand();
		optimizer.step(feat, targetNet.forward(feat));
	}
	
	trainNet.batch(100);
	critic.batch(100);
	NNHardAssert(critic.forward(trainNet.forward(testFeat), testLab) < 25, "SGD failed!");
	
	// MARK: Concat test
	
	Linear<> *comp1 = new Linear<>(10, 5);
	Sequential<> *comp2 = new Sequential<>(new Linear<>(10, 10), new TanH<>(), new Linear<>(10, 25), new TanH<>());
	TanH<> *comp3 = new TanH<>(10);
	
	Tensor<double> inMat = Tensor<double>(10).rand().resize(1, 10);
	Tensor<double> outMat(1, 5 + 25 + 10);
	outMat.sub({ {}, {  0, 5 } }).copy(comp1->forward(inMat));
	outMat.sub({ {}, {  5, 25 } }).copy(comp2->forward(inMat));
	outMat.sub({ {}, { 30, 10 } }).copy(comp3->forward(inMat));
	
	Concat<> concat(comp1, comp2, comp3);
	concat.forward(inMat);
	
	NNHardAssert(MSE<>(concat.outputs()).forward(outMat, concat.output()) < 1e-9, "Concat::forward failed!");
	
	Tensor<double> blam1 = Tensor<double>(comp1->output().shape(), true).rand();
	Tensor<double> blam2 = Tensor<double>(comp2->output().shape(), true).rand();
	Tensor<double> blam3 = Tensor<double>(comp3->output().shape(), true).rand();
	Tensor<double> inGrad2(inMat.shape(), true);
	inGrad2.addMM(comp1->backward(inMat, blam1));
	inGrad2.addMM(comp2->backward(inMat, blam2));
	inGrad2.addMM(comp3->backward(inMat, blam3));
	
	Tensor<double> blam = Tensor<double>::flatten({ &blam1, &blam2, &blam3 }).resize(1, 40);
	concat.backward(inMat, blam);
	
	NNHardAssert(MSE<>(inMat.shape()).forward(inGrad2, concat.inGrad()) < 1e-9, "Concat::backward failed!");
}

Tensor<> extrapolate(Sequencer<> &model, const Tensor<> &context, size_t length)
{
	size_t sequenceLength = model.sequenceLength();
	size_t bats = model.batch();
	
	model.forget();
	model.sequenceLength(1);
	model.batch(1);
	
	for(size_t i = 0; i < context.size(0); ++i)
	{
		model.forward(context.narrow(0, i));
	}
	
	Tensor<> result(length, 1, 1);
	for(size_t i = 0; i < length; ++i)
	{
		result.narrow(0, i).copy(model.forward(model.output()));
	}
	
	model.sequenceLength(sequenceLength);
	model.batch(bats);
	
	return result;
}

void plot(const string &filename, Sequencer<> &model, const Tensor<> &train, const Tensor<> &test)
{
	Tensor<> preds = extrapolate(model, train.reshape(train.size(0), 1, 1), test.size(0));
	Tensor<> full(train.size(0) + test.size(0), 3);
	full.fill(File<>::unknown);
	full.sub({ { 0, train.size(0) }, { 0 } }).copy(train.reshape(train.size(0), 1));
	full.sub({ { train.size(0), test.size(0) }, { 1 } }).copy(test.reshape(test.size(0), 1));
	full.sub({ { train.size(0), test.size(0) }, { 2 } }).copy(preds);
	File<>::saveArff(full, filename);
}

void testRNN()
{
	Tensor<> data(500);
	for(size_t i = 0; i < data.size(0); ++i)
		data(i) = sin(0.1 * i);
	data.normalize();
	
	size_t seqs = 100;
	size_t bats = 1;
	size_t epochs = 1000;
	double validation = 0.33;
	double learningRate = 0.0001;
	
	Sequencer<> rnn(
		new Sequential<>(
			new Linear<>(1, 32),
			new TanH<>(),
			new Linear<>(32),
			new TanH<>(),
			new Linear<>(1)
		),
		seqs,
		bats
	);
	MSE<> critic(rnn.outputs(), bats);
	SGD<> optimizer(rnn, critic);
	optimizer.learningRate(learningRate);
	
	Tensor<> train = data.narrow(0, 0, (1.0 - validation) * data.size(0));
	Tensor<> test = data.narrow(0, train.size(0), data.size(0) - train.size(0));
	
	Tensor<> trainFeat = train.narrow(0, 0, train.size(0) - 1);
	Tensor<> trainLab = train.narrow(0, 1, train.size(0) - 1);
	
	Tensor<> preds = extrapolate(rnn, train.resize(train.size(0), 1, 1), test.size(0));
	cout << "initial error: " << MSE<>(preds.size(0), false).forward(preds, test.resize(test.size(0), 1, 1)) << endl;
	
	SequenceBatcher<> batcher(trainFeat.resize(trainFeat.size(0), 1), trainLab.resize(trainLab.size(0), 1), seqs, bats);
	
	for(size_t i = 0; i < epochs; ++i)
	{
		batcher.reset();
		optimizer.step(batcher.features(), batcher.labels());
		
		preds = extrapolate(rnn, train.resize(train.size(0), 1, 1), test.size(0));
		
		Progress<>::display(i, epochs);
		
		cout << "\terror: " << MSE<>(preds.size(0), false).forward(preds, test.resize(test.size(0), 1, 1)) << flush;
	}
	Progress<>::display(epochs, epochs, '\n');
	
	preds = extrapolate(rnn, train.resize(train.size(0), 1, 1), test.size(0));
	cout << "final error: " << MSE<>(preds.size(0), false).forward(preds, test.resize(test.size(0), 1, 1)) << endl;
	
	plot("plot.arff", rnn, train, test);
}

int main()
{
	cout << "===== Testing Tensor =====" << endl;
	testTensor();
	cout << "Tensor test passed!" << endl << endl;
	
	cout << "===== Testing Algebra =====" << endl;
	testAlgebra();
	cout << "Algebra test passed!" << endl << endl;
	
	cout << "===== Testing Neural Networks =====" << endl;
	testNeuralNet();
	cout << "Neural networks test passed!" << endl << endl;
	
	cout << "===== Testing Recurrent Neural Networks =====" << endl;
	testRNN();
	cout << "Recurrent neural networks test passed!" << endl << endl;
	
	cout << "All unit tests passed!" << endl;
	
	return 0;
}
