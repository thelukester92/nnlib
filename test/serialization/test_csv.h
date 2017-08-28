#ifndef TEST_CSV_H
#define TEST_CSV_H

#include "nnlib/serialization/csv.h"
#include <sstream>
using namespace nnlib;

void TestCsvSerializer()
{
	SerializedNode data((SerializedNode::Array()));
	
	data.append(std::vector<double>{ 1, 2, 3, 4, 5 });
	data.append(std::vector<int>{ 6, 7, 8, 9, 0 });
	data.append(std::vector<std::string>{ "this", "is", "a", "te\"st" });
	
	CsvSerializer::write(data, std::cout);
	
	std::string s = "1,2,3\n\"fo, ur\",5.0,six\n7\"\"";
	std::stringstream ss(s);
	SerializedNode node = CsvSerializer::read(ss);
	CsvSerializer::write(node, std::cout);
}

#endif
