#ifndef TEST_SERIALIZATION_BINARY_H
#define TEST_SERIALIZATION_BINARY_H

#include "nnlib/serialization/binary.h"
#include "nnlib/tensor.h"
#include "test_archive.h"
using namespace nnlib;

void TestBinaryArchive()
{
	TestArchive<BinaryInputArchive, BinaryOutputArchive>();
	
	Tensor<> test = Tensor<>(3, 4, 5).rand();
	Tensor<> test2;
	
	{
		BinaryOutputArchive out("bin/.tmp");
		out(test);
	}
	
	{
		BinaryInputArchive in("bin/.tmp");
		in(test2);
	}
	
	for(auto i = test.begin(), j = test2.begin(), k = test.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "BinaryArchive serialization failed!");
}

#endif
