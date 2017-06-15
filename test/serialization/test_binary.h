#ifndef TEST_SERIALIZATION_BINARY_H
#define TEST_SERIALIZATION_BINARY_H

#include "nnlib/serialization/binary.h"
#include "test_archive.h"
using namespace nnlib;

void TestBinaryArchive()
{
	TestArchive<BinaryInputArchive, BinaryOutputArchive>();
}

#endif
