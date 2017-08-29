#ifndef TEST_JSON_SERIALIZER_H
#define TEST_JSON_SERIALIZER_H

#include "nnlib/serialization/jsonserializer.h"
#include "nnlib/serialization/serialized.h"

#include "nnlib/nn/linear.h"
#include "nnlib/nn/sequential.h"
#include "nnlib/nn/tanh.h"

using namespace nnlib;

void TestJSONSerializer()
{
	// basic serialization
	
	{
		Serialized s;
		s.set("library", "nnlib");
		s.set("awesome", true);
		s.set("number", 42);
		
		std::stringstream ss;
		JSONSerializer::write(s, ss);
		
		Serialized d = JSONSerializer::read(ss);
		
		NNAssertEquals(d.get<std::string>("library"), "nnlib", "JSONSerializer failed!");
		NNAssertEquals(d.get<bool>("awesome"), true, "JSONSerializer failed!");
		NNAssertEquals(d.get<size_t>("number"), 42, "JSONSerializer failed!");
	}
	
	// neural network serialization
	
	{
		Sequential<> nn(
			new Linear<>(10, 5),
			new TanH<>(),
			new Linear<>(10),
			new TanH<>()
		);
		
		std::stringstream ss;
		JSONSerializer::write(nn, ss);
		
		Sequential<> *deserialized = JSONSerializer::read(ss).as<Sequential<> *>();
		
		auto &p1 = nn.parameters();
		auto &p2 = deserialized->parameters();
		
		for(auto i = p1.begin(), j = p2.begin(), end = p1.end(); i != end; ++i, ++j)
		{
			NNAssertAlmostEquals(*i, *j, 1e-12, "JSONSerializer failed!");
		}
		
		delete deserialized;
	}
}

#endif
