#ifndef TEST_CSV_H
#define TEST_CSV_H

#include "nnlib/serialization/csv.h"
using namespace nnlib;

void TestCsvSerializer()
{
	SerializedNode data((SerializedNode::Array()));
	
	std::vector<double> a({ 1, 2, 3, 4, 5 });
	std::vector<double> b({ 6, 7, 8, 9, 0 });
	
	data.append(a.begin(), a.end());
	data.append(b.begin(), b.end());
	
	CsvSerializer::write(data, std::cout);
}

#endif
