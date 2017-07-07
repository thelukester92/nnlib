#ifndef TEST_SERIALIZATION_ARFF_H
#define TEST_SERIALIZATION_ARFF_H

#include "nnlib/serialization/arff.h"
#include <sstream>
using namespace nnlib;

void TestArffSerializer()
{
	Tensor<> tensor1, tensor2, tensor3;
	std::stringstream ss1, ss2, ss3;
	
	ss1
		<< "@relation 'this is a test'\n"
		<< "% this is a comment\n"
		<< "@attribute attr1 numeric\n"
		<< "@attribute 'attribute 2' integer\n"
		<< "@attribute \"the \\\"third\\\" attr\" real\n"
		<< "@attribute four NUMERiC\n"
		<< "@attribute five INTEGER\n"
		<< "@attribute six real\n"
		<< "@attribute sev real\n"
		<< "@attribute eig real\n"
		<< "@attribute nin real\n"
		<< "@attribute ten {this,'is an',enum}\n"
		<< "@data\n"
		<< "8,6,7,5,3,0,9,1.0,3.14,this\n"
		<< "1,2,3,4,5,6,7,8.1,4.22,\"is an\"\n"
		<< "1,2,3,4,5,6,7,8.1,4.22, 'enum'\n";
	
	tensor3.resize(3, 10).copy({
		8, 6, 7, 5, 3, 0, 9, 1.0, 3.14, 0,
		1, 2, 3, 4, 5, 6, 7, 8.1, 4.22, 1,
		1, 2, 3, 4, 5, 6, 7, 8.1, 4.22, 2
	});
	
	auto relation = ArffSerializer::read(tensor1, ss1);
	ArffSerializer::write(tensor1, ss2, relation);
	ArffSerializer::read(tensor2, ss2);
	
	NNAssertEquals(tensor1.shape(), tensor3.shape(), "ArffSerializer::read failed! Wrong shape.");
	for(auto i = tensor1.begin(), j = tensor3.begin(), k = tensor1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "ArffSerializer::read failed! Wrong data.");
	
	NNAssertEquals(tensor1.shape(), tensor2.shape(), "ArffSerializer::write failed! Wrong shape.");
	for(auto i = tensor1.begin(), j = tensor2.begin(), k = tensor1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "ArffSerializer::write failed! Wrong data.");
	
	
	ArffSerializer::write(tensor1, ss3);
	ArffSerializer::read(tensor2, ss3);
	
	NNAssertEquals(tensor1.shape(), tensor2.shape(), "ArffSerializer::write without Relation failed! Wrong shape.");
	for(auto i = tensor1.begin(), j = tensor2.begin(), k = tensor1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "ArffSerializer::write without Relation failed! Wrong data.");
	
	bool ok = false;
	try
	{
		std::stringstream ss4("");
		ArffSerializer::read(tensor1, ss4);
	}
	catch(const Error &e)
	{
		ok = true;
	}
	NNAssert(ok, "ArffSerializer::read did not throw a meaningful error on an intentionally bad stream!");
}

#endif
