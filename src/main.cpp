#include "tensor.h"
#include "random.h"
#include "error.h"
#include <iostream>
using namespace nnlib;
using namespace std;

int main()
{
	Tensor<double> t(5), u(5), v(5);
	Random r;
	
	for(size_t i = 0; i < t.size(); ++i)
	{
		t[i] = r.normal(0.0, 1.0, 3.0);
		u[i] = r.normal(0.0, 1.0, 3.0);
	}
	
	v = t + u;
	
	for(size_t i = 0; i < v.size(); ++i)
		Assert(v[i] == t[i] + u[i], "Addition failed!");
	
	cout << "Passed all tests!" << endl;
	
	return 0;
}
