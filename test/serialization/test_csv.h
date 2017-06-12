#ifndef TEST_SERIALIZATION_CSV_H
#define TEST_SERIALIZATION_CSV_H

#include "nnlib/serialization/csv.h"
#include <sstream>
using namespace nnlib;

void TestCSVArchive()
{
	Tensor<> tensor, tensor2;
	std::stringstream ss;
	
	{
		ss << "8,6,7,5,3,0,9,1.0,3.14,6.28\n";
		ss << "1,2,3,4,5,6,7,8.1,4.22,3.14\n";
	}
	
	{
		CSVInputArchive in(ss);
		in(tensor);
	}
	
	std::stringstream ss2;
	
	{
		CSVOutputArchive out(ss2);
		out(tensor);
		
		CSVInputArchive in(ss2);
		in(tensor2);
	}
	
	for(auto i = tensor.begin(), j = tensor2.begin(), k = tensor.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "CSVArchive failed!");
}

#endif
