#include <iostream>
#include <vector>
#include "vector.h"
#include "matrix.h"
using namespace std;
using namespace nnlib;

int main()
{
	Vector<double> vec(10);
	vec(4) = 3.14;
	
	for(auto val : vec)
		cout << val << " ";
	cout << endl;
	
	Matrix<double> mat(3, 3);
	mat.fill(3.14);
	mat(2, 1) = 1.0;
	
	Vector<double> row = mat[1];
	row.fill(5);
	
	for(auto val : mat)
		cout << val << " ";
	cout << endl;
	
	return 0;
}
