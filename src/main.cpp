#include <iostream>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

void testVector()
{
	Vector<> v(2, 3.14);
	NNHardAssert(v(1) == 3.14, "Vector::Vector(size_t, double) failed!");
	
	Vector<> u = v;
	NNHardAssert(u(1) == 3.14, "Vector::Vector(Vector &) failed!");
}

void testNeuralNet()
{
	Sequential neuralDecomposition(
		new Concat(
			new Sequential(
				new Linear(100),
				new Activation<Sin>()
			),
			new Sequential(
				new Linear(10),
				new Activation<Tanh>()
			),
			new Linear(10)
		),
		new Linear(10)
	);
}

int main()
{
	testVector();
	testNeuralNet();
	return 0;
}
