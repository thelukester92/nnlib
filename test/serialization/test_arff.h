#ifndef TEST_SERIALIZATION_ARFF_H
#define TEST_SERIALIZATION_ARFF_H

#include "nnlib/serialization/arff.h"
#include <sstream>
using namespace nnlib;

void TestArffSerializer()
{
	Tensor<> tensor1, tensor2, tensor3;
	std::stringstream ss1, ss2;
	
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
		<< "@attribute ten real\n"
		<< "@data\n"
		<< "8,6,7,5,3,0,9,1.0,3.14,6.28\n"
		<< "1,2,3,4,5,6,7,8.1,4.22,3.14\n";
	
	tensor3.resize(2, 10).copy({
		8, 6, 7, 5, 3, 0, 9, 1.0, 3.14, 6.28,
		1, 2, 3, 4, 5, 6, 7, 8.1, 4.22, 3.14
	});
	
	auto relation = ArffSerializer::read(tensor1, ss1);
	std::cout << "deserialized to " << std::endl;
	std::cout << tensor1 << std::endl;
	
	ArffSerializer::write(tensor1, ss2, relation);
	std::cout << "reserialized to " << std::endl;
	std::cout << ss2.str() << std::endl;
	
	ArffSerializer::read(tensor2, ss2);
	
	NNAssertEquals(tensor1.shape(), tensor3.shape(), "ARFFUtil::read failed! Wrong shape.");
	for(auto i = tensor1.begin(), j = tensor3.begin(), k = tensor1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "ARFFUtil::read failed! Wrong data.");
	
	NNAssertEquals(tensor1.shape(), tensor2.shape(), "ARFFUtil::write failed! Wrong shape.");
	for(auto i = tensor1.begin(), j = tensor2.begin(), k = tensor1.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "ARFFUtil::write failed! Wrong data.");
	
	/// \todo metadata
}

#endif
