#include <iostream>
#include "nnlib.h"
using namespace std;
using namespace nnlib;

void testVector()
{
	Vector<> v(2, 3.14);
	NNHardAssert(v(1) == 3.14, "Vector::Vector(size_t, double) failed!");
	
	Vector<> u = v;
	NNHardAssert(u(1) == 3.14, "Vector::Vector(Vector &) failed!")
	
	v.resize(4);
}

int main()
{
	testVector();
	return 0;
}
