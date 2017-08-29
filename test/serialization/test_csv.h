#ifndef TEST_CSV_H
#define TEST_CSV_H

#include "nnlib/serialization/csv.h"
#include <sstream>
using namespace nnlib;

void TestCsvSerializer()
{
	/*
	Serialized data(Serialized::Array);
	
	data.add(std::vector<double>{ 1, 2, 3, 4, 5 });
	data.add(std::vector<int>{ 6, 7, 8, 9, 0 });
	data.add(std::vector<std::string>{ "this", "is", "a", "te\"st" });
	
	CsvSerializer::write(data, std::cout);
	*/
	
	std::string s = "1,2,3\n\"fo, ur\",5.0,six\n7\"\"";
	std::stringstream ss(s);
	Serialized node = CsvSerializer::read(ss);
	CsvSerializer::write(node, std::cout);
}

#endif
