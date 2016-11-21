#include "tensor.h"
#include "error.h"
#include "random.h"
#include "loader.h"
#include "linear.h"
#include "tanh.h"
#include "sequential.h"
#include "squared_error.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
using namespace nnlib;
using namespace std;

void testTensor();
void testCorrectness();
double testEfficiency(size_t inps, size_t outs, size_t epochs, function<void()> &start, function<void()> &end);
void testLine();
void testMNIST(function<void()> &start, function<void()> &end);

int main()
{
	size_t inps				= 10000;
	size_t outs				= 1000;
	size_t epochs			= 100;
	
	using clock = chrono::high_resolution_clock;
	clock::time_point start;
	function<void()> startFn = [&](void) { start = clock::now(); };
	function<void()> endFn = [&](void) { cout << "took " << chrono::duration<double>(clock::now() - start).count() << " seconds" << endl; };
	
	try
	{
		testTensor();
		testCorrectness();
		testEfficiency(inps, outs, epochs, startFn, endFn);
		testLine();
		testMNIST(startFn, endFn);
	}
	catch(exception &e)
	{
		cerr << "A test failed! " << e.what() << endl;
		return 1;
	}
	cout << "All tests passed!" << endl;
	return 0;
}

void testTensor()
{
	Vector<size_t> basicTensor(5);
	basicTensor.fill(0);
	
	vector<size_t> vec(5);
	fill(vec.begin(), vec.end(), 0);
	
	for(size_t i = 0; i < basicTensor.size(); ++i)
		NNLibAssert(basicTensor[i] == vec[i], "Fill failed!");
	
	for(size_t i = 0; i < basicTensor.size(); ++i)
		basicTensor[i] = vec[i] = i;
	
	for(size_t i = 0; i < basicTensor.size(); ++i)
		NNLibAssert(basicTensor[i] == vec[i], "Setting via operator[] failed!");
	
	Matrix<double> matrix(2, 3);
	matrix(0, 0) = 1;
	matrix(0, 1) = 2;
	matrix(0, 2) = 3;
	matrix(1, 0) = 4;
	matrix(1, 1) = 5;
	matrix(1, 2) = 6;
	matrix *= 3.14;
	
	for(size_t i = 0; i < matrix.size(); ++i)
		NNLibAssert(matrix[i] == (i + 1) * 3.14, "Matrix scaling failed!");
	
	Matrix<double> A(3, 3), B(3, 3);
	for(size_t i = 0; i < A.size(); ++i)
		A[i] = i, B[i] = B.size() - i;
	
	Matrix<double> C = A + B;
	NNLibAssert(A.size() == C.size(), "Matrix assignment failed!");
	for(size_t i = 0; i < C.size(); ++i)
		NNLibAssert(C[i] == A[i] + B[i], "Matrix addition failed!");
	
	Matrix<double> D = A - B;
	NNLibAssert(A.size() == D.size(), "Matrix assignment failed!");
	for(size_t i = 0; i < D.size(); ++i)
		NNLibAssert(D[i] == A[i] - B[i], "Matrix subtraction failed!");
	
	Matrix<double> E = -A;
	NNLibAssert(A.size() == E.size(), "Matrix assignment failed!");
	for(size_t i = 0; i < E.size(); ++i)
		NNLibAssert(E[i] == -A[i], "Matrix negation failed!");
	
	Matrix<double> X = Matrix<double>::identity(2, 2);
	Matrix<double> Y(2, 3);
	Y(0, 0) = 8;
	Y(0, 1) = 6;
	Y(0, 2) = 7;
	Y(1, 0) = 5;
	Y(1, 1) = 3;
	Y(1, 2) = 0;
	
	Matrix<double> Z = X * Y;
	NNLibAssert(Z.rows() == X.rows() && Z.cols() == Y.cols(), "Matrix multiplication failed!");
	for(size_t i = 0; i < Z.rows(); ++i)
		for(size_t j = 0; j < Z.cols(); ++j)
			NNLibAssert(Z(i, j) == Y(i, j), "Identity matrix multiplication failed!");
	
	X(0, 0) = 3;
	X(0, 1) = 2;
	X(1, 0) = 9;
	X(1, 1) = -7;
	Z -= X * Y;
	
	Matrix<double> Q(2, 3);
	Q(0, 0) = 34;
	Q(0, 1) = 24;
	Q(0, 2) = 21;
	Q(1, 0) = 37;
	Q(1, 1) = 33;
	Q(1, 2) = 63;
	
	for(size_t i = 0; i < Z.rows(); ++i)
		for(size_t j = 0; j < Z.cols(); ++j)
			NNLibAssert(Z(i, j) == Y(i, j) - Q(i, j), "Matrix multiplication+subtraction failed!");
	
	Matrix<double> T = ~Q;
	NNLibAssert(T.rows() == Q.cols() && T.cols() == Q.rows(), "Matrix transposition failed!");
	for(size_t i = 0; i < T.rows(); ++i)
		for(size_t j = 0; j < T.cols(); ++j)
			NNLibAssert(T(i, j) == Q(j, i), "Matrix transposition failed!");
	
	Matrix<double> U(0, 0);
	U = Matrix<double>::identity(3, 3) * ~Q;
	NNLibAssert(T.rows() == U.rows() && T.cols() == U.cols(), "Matrix transposition failed!");
	for(size_t i = 0; i < U.rows(); ++i)
		for(size_t j = 0; j < U.cols(); ++j)
			NNLibAssert(U(i, j) == T(i, j), "Matrix transposition failed!");
	
	Matrix<double> W = X * Y - Z;
	Vector<double> a(5), b(5);
	Vector<double> c = a - b;
}

void testCorrectness()
{
	size_t inps = 2, outs = 3;
	Linear<double> layer(inps, outs);
	Matrix<double> input(1, inps), target(1, outs);
	
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
	
	input[0] = 3.14;
	input[1] = 10.0;
	
	target[0] = 3.14;
	target[1] = 11.0;
	target[2] = 15.14;
	
	Matrix<double> &result = layer.forward(input);
	for(size_t i = 0; i < outs; ++i)
		NNLibAssert(result[i] == target[i], "forward failed!");
	
	Matrix<double> &blame = layer.backward(input, target);
	NNLibAssert(blame[0] == 18.28 && blame[1] == 26.14, "backward failed to assign correct input blame!");
	
	Vector<double> &biasBlame = *(Vector<double> *)layer.blame()[1];
	for(size_t i = 0; i < outs; ++i)
		NNLibAssert(biasBlame(i) == target[i], "backward failed to assign correct bias blame!");
	
	Matrix<double> expectedBlame(outs, inps);
	expectedBlame[0] = 9.8596;
	expectedBlame[1] = 31.4;
	expectedBlame[2] = 34.54;
	expectedBlame[3] = 110.0;
	expectedBlame[4] = 47.5396;
	expectedBlame[5] = 151.4;
	
	Matrix<double> &weightsBlame = *(Matrix<double> *)layer.blame()[0];
	for(size_t i = 0; i < expectedBlame.size(); ++i)
		NNLibAssert(fabs(weightsBlame[i] - expectedBlame[i]) < 1e-12, "backward failed to assign correct weights blame!");
	
	Tanh<double> activation(outs);
	Matrix<double> &act = activation.forward(layer.output());
	for(size_t i = 0; i < act.size(); ++i)
		NNLibAssert(fabs(act[i] - tanh(layer.output()[i])) < 1e-9, "tanh forward failed!");
	
	Matrix<double> &tanhBlame = activation.backward(layer.output(), target);
	for(size_t i = 0; i < tanhBlame.size(); ++i)
		NNLibAssert(tanhBlame[i] == target[i] * (1.0 - act[i] * act[i]), "tanh backward failed!");
	
	Sequential<double> nn;
	nn.add(&layer);
	nn.add(&activation);
	
	try
	{
		NNLibAssert(layer.inputBlame().begin() == nn.inputBlame().begin(), "Sequential failed to share input blame buffer!");
		NNLibAssert(activation.output().begin() == nn.output().begin(), "Sequential failed to share output buffer!");
		
		Matrix<double> tanOut;
		tanOut.copy(activation.forward(layer.forward(input)));
		
		Matrix<double> seqOut;
		seqOut.copy(nn.forward(input));
		
		for(size_t i = 0; i < seqOut.size(); ++i)
			NNLibAssert(tanOut[i] == seqOut[i], "sequential forward failed!");
		
		Matrix<double> layIn;
		layIn.copy(layer.backward(input, activation.backward(layer.output(), target)));
		
		Matrix<double> seqIn;
		seqIn.copy(nn.backward(input, target));
		
		for(size_t i = 0; i < seqIn.size(); ++i)
			NNLibAssert(layIn[i] == seqIn[i], "sequential backward failed!");
	}
	catch(runtime_error &e)
	{
		nn.release(1);
		nn.release(0);
		throw e;
	}
	
	// release the layers, since they weren't dynamically allocated!
	nn.release(1);
	nn.release(0);
}

double testEfficiency(size_t inps, size_t outs, size_t epochs, function<void()> &start, function<void()> &end)
{
	Matrix<double> weights(outs, inps);
	Matrix<double> input(1, inps), result(1, outs);
	Vector<double> bias(outs);
	Random r;
	
	r.fillNormal(weights);
	r.fillNormal(bias);
	r.fillNormal(input);
	result.fill(0.0);
	
	start();
	for(size_t i = 0; i < epochs; ++i)
		result += input * ~weights + bias;
	end();
	
	double resultSum = 0.0;
	for(size_t i = 0; i < outs; ++i)
		resultSum += result[i];
	
	return resultSum;
}

void testLine()
{
	Sequential<double> nn;
	nn.add(new Linear<double>(1, 1), new Tanh<double>(1));
	
	Matrix<double> data(1000, 1), lab(1000, 1);
	for(size_t i = 0; i < data.size(); ++i)
	{
		data[i] = i / 1000.0;
		lab[i] = i / 1000.0;
	}
	
	Vector<double> param = Vector<double>::flatten(nn.parameters());
	Vector<double> blame = Vector<double>::flatten(nn.blame());
	
	Random r;
	r.fillNormal(param, 0.0, 0.03, 0.1);
	
	RandomIterator ri(1000);
	for(size_t i = 0; i < 100; ++i)
	{
		ri.reset();
		for(auto i : ri)
		{
			nn.forwardSingle(data.row(i));
			nn.backwardSingle(data.row(i), lab.row(i) - nn.output().row(0));
			
			auto p = param.begin();
			auto b = blame.begin();
			for(; p != param.end(); ++p, ++b)
				*p += 0.01 * *b;
		}
	}
	
	double sse = 0;
	for(size_t i = 0; i < 1000; ++i)
	{
		Vector<double> _row = data.row(i);
		Matrix<double> row(_row, 1, data.cols());
		nn.forward(row);
		sse += (lab(i, 0) - nn.output()[0]) * (lab(i, 0) - nn.output()[0]);
	}
	NNAssert(sse < 5, "Linear regression failed!");
}

void testMNIST(function<void()> &start, function<void()> &end)
{
/*
	cout << "==================== MNIST ====================" << endl;
	
	Sequential<double> nn;
	nn.add(
		new Linear<double>(784, 80), new Tanh<double>(80),
		new Linear<double>(80, 30), new Tanh<double>(30),
		new Linear<double>(30, 10), new Tanh<double>(10)
	);
	SquaredError<double> critic(10);
	
	Matrix<double> train = Loader<double>::load("../datasets/mnist/train.raw");
	Matrix<double> test = Loader<double>::load("../datasets/mnist/test.raw");
	
	Matrix<double> trainFeat(train.rows(), train.cols() - 1), trainLab(train.rows(), 10);
	Matrix<double> testFeat(test.rows(), test.cols() - 1), testLab(test.rows(), 10);
	
	trainLab.fill(0);
	for(size_t i = 0; i < train.rows(); ++i)
	{
		for(size_t j = 0; j < trainFeat.cols(); ++j)
			trainFeat(i, j) = train(i, j) / 255.0;
		trainLab(i, train(i, trainFeat.cols())) = 1;
	}
	
	testLab.fill(0);
	for(size_t i = 0; i < test.rows(); ++i)
	{
		for(size_t j = 0; j < testFeat.cols(); ++j)
			testFeat(i, j) = test(i, j) / 255.0;
		testLab(i, test(i, testFeat.cols())) = 1;
	}
	
	Vector<double> parameters = Vector<double>::flatten(nn.parameters());
	Vector<double> blame = Vector<double>::flatten(nn.blame());
	
	Random r;
	for(size_t i = 0; i < nn.modules(); ++i)
		for(auto j : nn.module(i).parameters())
			r.fillNormal(*j, 0.0, std::max(0.03, 1.0 / nn.module(i).inputCount()));
	
	RandomIterator ri(train.rows());
	Vector<double> feat, lab;
	double learningRate = 1e-2;
	
	size_t misclassified = 0;
	for(size_t k = 0; k < test.rows(); ++k)
	{
		testFeat.row(k, feat);
		testLab.row(k, lab);
		nn.forward(feat);
		size_t indexOfMax = 0, actualMax = 0;
		for(size_t l = 1; l < 10; ++l)
		{
			if(nn.output()(l) > nn.output()(indexOfMax))
				indexOfMax = l;
			if(lab[l] > lab[actualMax])
				actualMax = l;
		}
		if(indexOfMax != actualMax)
			++misclassified;
	}
	cout << "Begin: " << misclassified << endl;
	
	auto pe = parameters.end();
	
	start();
	for(size_t epoch = 0; epoch < 1; ++epoch)
	{
		ri.reset();
		for(auto i : ri)
		{
			trainFeat.row(i, feat);
			trainLab.row(i, lab);
			
			nn.forward(feat);
			nn.backward(feat, critic.backward(nn.output(), lab));
			
			auto p = parameters.begin();
			auto b = blame.begin();
			
			for(; p != pe; ++p, ++b)
				*p += learningRate * *b;
		}
	}
	end();
	
	misclassified = 0;
	for(size_t k = 0; k < test.rows(); ++k)
	{
		testFeat.row(k, feat);
		testLab.row(k, lab);
		nn.forward(feat);
		size_t indexOfMax = 0, actualMax = 0;
		for(size_t l = 1; l < 10; ++l)
		{
			if(nn.output()(l) > nn.output()(indexOfMax))
				indexOfMax = l;
			if(lab[l] > lab[actualMax])
				actualMax = l;
		}
		if(indexOfMax != actualMax)
			++misclassified;
	}
	cout << "End: " << misclassified << "     " << endl;
*/
}
