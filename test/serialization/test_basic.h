#ifndef TEST_SERIALIZATION_BASIC_H
#define TEST_SERIALIZATION_BASIC_H

#include "nnlib/serialization/basic.h"
#include "test_archive.h"
#include <sstream>
using namespace nnlib;

void TestBasicArchive()
{
	TestArchive<BasicInputArchive, BasicOutputArchive>();
}

#endif
