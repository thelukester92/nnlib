#ifndef TEST_SERIALIZATION_CSV_H
#define TEST_SERIALIZATION_CSV_H

#include "nnlib/serialization/csv.h"
#include <sstream>
using namespace nnlib;

void TestCsvSerializer()
{
	Tensor<> tensor1, tensor2, tensor3;
	std::stringstream ss1, ss2;
	
	ss1 << "\n\n"
		<< "8,\t6,7,5,3,0, 9,1.0,3.14,6.28\n\n"
		<< "1,2, 3,4,5,6,7,8.1,\t4.22,3.14\n\n";
	
	tensor3.resize(2, 10).copy({
		8, 6, 7, 5, 3, 0, 9, 1.0, 3.14, 6.28,
		1, 2, 3, 4, 5, 6, 7, 8.1, 4.22, 3.14
	});
	
	CsvSerializer::read(tensor1, ss1);
	CsvSerializer::write(tensor1, ss2);
	CsvSerializer::read(tensor2, ss2);
	
	NNAssertEquals(tensor1.shape(), tensor3.shape(), "CsvSerializer::read failed! Wrong shape.");
	for(auto i = tensor1.begin(), j = tensor3.begin(), k = tensor1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "CsvSerializer::read failed! Wrong data.");
	
	NNAssertEquals(tensor1.shape(), tensor2.shape(), "CsvSerializer::write failed! Wrong shape.");
	for(auto i = tensor1.begin(), j = tensor2.begin(), k = tensor1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "CsvSerializer::write failed! Wrong data.");
}

#endif
