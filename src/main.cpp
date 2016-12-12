#include <iostream>
#include <vector>
#include "vector.h"
#include "matrix.h"
using namespace std;
using namespace nnlib;

int main()
{
	Matrix<double> A(3, 3), B(3, 4);
	
	for(size_t i = 0; i < A.rows(); ++i)
		for(size_t j = 0; j < A.cols(); ++j)
			A(i, j) = i == j ? i : 0;
	
	B.fill(3.14);
	
	Matrix<double> C(3, 4);
	Matrix<double>::multiply(A, B, C);
	
	for(size_t i = 0; i < C.rows(); ++i)
	{
		for(size_t j = 0; j < C.cols(); ++j)
			cout << C(i, j) << "\t";
		cout << endl;
	}
	
	return 0;
}
