#ifndef TEST_SERIALIZATION_BASIC_H
#define TEST_SERIALIZATION_BASIC_H

#include "nnlib/serialization/basic.h"
#include <sstream>
using namespace nnlib;

void TestBasicArchive()
{
	std::stringstream ss;
	BasicInputArchive in(ss);
	BasicOutputArchive out(ss);
	/// \todo fill me in
}

#endif
