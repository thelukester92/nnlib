#ifndef TEST_JSON_SERIALIZER_H
#define TEST_JSON_SERIALIZER_H

#include "nnlib/serialization/jsonserializer.h"
#include "nnlib/serialization/serialized.h"
using namespace nnlib;

void TestJSONSerializer()
{
	std::string s = "{ \"library\": \"nnlib\", \"awesome\": true, \"number\": 42 }";
	Serialized d = JSONSerializer::readString(s);
	JSONSerializer::write(d, std::cout);
}

#endif
