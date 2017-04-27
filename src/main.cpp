#include <iostream>
#include <math.h>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

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
	
	MSE<> critic(trainNet);
	auto optimizer = makeOptimizer<SGD>(trainNet, critic);
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
}

void testMNIST()
{
	cout << "Setting up..." << endl;
	
	Relation rel;
	Tensor<double> train = File<>::loadArff("../data/mnist.train.arff", &rel);
	Tensor<double> test = File<>::loadArff("../data/mnist.test.arff");
	size_t outs = rel.attrVals(rel.size() - 1).size();
	
	Tensor<double> trainFeat = train.sub({ {}, { 0, train.size(1) - 1 } }).scale(1.0 / 255.0);
	Tensor<double> trainLab = train.sub({ {}, { train.size(1) - 1 } });
	
	Tensor<double> testFeat = test.sub({ {}, { 0, test.size(1) - 1 } }).scale(1.0 / 255.0);
	Tensor<double> testLab = test.sub({ {}, { test.size(1) - 1 } });
	
	Sequential<> nn(
		new Linear<>(trainFeat.size(1), 300), new TanH<>(),
		new Linear<>(100), new TanH<>(),
		new Linear<>(outs), new TanH<>(),
		new LogSoftMax<>()
	);
	NLL<> critic(nn);
	auto optimizer = makeOptimizer<RMSProp>(nn, critic).learningRate(0.001);
	
	Batcher<> batcher(trainFeat, trainLab, 100);
	nn.batch(batcher.batch());
	critic.batch(batcher.batch());
	
	cout << "Training..." << endl;
	
	size_t epochs = 10;
	size_t k = 0, tot = epochs * batcher.batches();
	for(size_t i = 0; i < epochs; ++i)
	{
		batcher.reset();
		do
		{
			Progress<>::display(k++, tot);
			optimizer.step(batcher.features(), batcher.labels());
		}
		while(batcher.next());
	}
	Progress<>::display(tot, tot, '\n');
}

#include <unistd.h>

int main()
{
	size_t steps = 1000;
	RandomEngine::seed(12345);
	
	Tensor<double> sequence(steps, 1, 1);
	for(size_t i = 0; i < steps; ++i)
	{
		sequence(i, 0, 0) = sin(0.1 * i);
	}
	
	Sequential<> rnn(
		new Recurrent<>(1, 10),
		new Linear<>(10, 1)
	);
	MSE<> critic(rnn);
	Tensor<double> params = Tensor<double>::flatten(rnn.parameters());
	Tensor<double> grads = Tensor<double>::flatten(rnn.grad());
	Tensor<double> state = Tensor<double>::flatten(rnn.innerState());
	Tensor<double> states(steps, state.size(0));
	
	state.fill(0);
	cout << "Before:" << endl;
	double foo = 0;
	for(size_t i = 0; i < steps - 1; ++i)
	{
		foo += critic.forward(rnn.forward(sequence.select(0, i)), sequence.select(0, i + 1));
	}
	cout << foo << endl;
	
	double lr = 0.1;
	for(size_t epoch = 0;; ++epoch)
	{
		state.fill(0);
		for(size_t i = 0; i < steps - 1; ++i)
		{
			rnn.forward(sequence.select(0, i));
			states.select(0, i).copy(state);
		}
		
		grads.fill(0);
		for(size_t i = steps - 1; i > 1; --i)
		{
			state.copy(states.select(0, i - 1));
			rnn.backward(sequence.select(0, i - 1), critic.backward(rnn.output(), sequence.select(0, i)));
		}
		params.addVV(grads, -lr);
		
		// state.copy(states.select(0, 0));
		// cout << rnn.output()(0, 0) << "\t" << sequence(0, 0, 0) << endl;
		
		state.fill(0);
		foo = 0;
		for(size_t i = 0; i < steps - 1; ++i)
		{
			foo += critic.forward(rnn.forward(sequence.select(0, i)), sequence.select(0, i + 1));
		}
		cout << "\rafter: " << setprecision(3) << fixed << foo << "\t" << epoch << flush;
	}
	cout << endl;
	
	return 0;
	
	
	cout << "===== Testing Tensor =====" << endl;
	testTensor();
	cout << "Tensor test passed!" << endl << endl;
	
	cout << "===== Testing Algebra =====" << endl;
	testAlgebra();
	cout << "Algebra test passed!" << endl << endl;
	
	cout << "===== Testing Neural Networks =====" << endl;
	testNeuralNet();
	cout << "Neural networks test passed!" << endl << endl;
	
	cout << "===== Testing on MNIST =====" << endl;
	testMNIST();
	cout << "MNIST test passed!" << endl << endl;
	
	return 0;
}
