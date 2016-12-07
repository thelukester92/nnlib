#include <iostream>
#include "vector.h"
using namespace std;
using namespace nnlib;

int main()
{
	Vector<double> vec(10);
	vec(4) = 3.14;
	
	for(auto val : vec)
		cout << val << " ";
	cout << endl;
	
	return 0;
}
