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
		s.set("notAwesome", false);
		s.set("number", 32);
		s.set("number", 42);
		s.set("nothing", Serialized::Null);
		s.set("emptyArray", Serialized::Array);
		s.set("emptyObject", Serialized::Object);
		s.set("nested", Serialized::Object);
		
		s.get<Serialized *>("nested")->set("indented", true);
		s.get<Serialized *>("nested")->set("powerLevel", "> 9000");
		
		// minified
		{
			std::stringstream ss;
			JSONSerializer::write(s, ss);
			
			Serialized d = JSONSerializer::read(ss);
			
			NNAssertEquals(d.get<std::string>("library"), "nnlib", "JSONSerializer failed!");
			NNAssertEquals(d.get<bool>("awesome"), true, "JSONSerializer failed!");
			NNAssertEquals(d.get<bool>("notAwesome"), false, "JSONSerializer failed!");
			NNAssertEquals(d.get<size_t>("number"), 42, "JSONSerializer failed!");
			NNAssertAlmostEquals(d.get<float>("number"), 42.0, 1e-12, "JSONSerializer failed!");
			NNAssertEquals(d.type("nothing"), Serialized::Null, "JSONSerializer failed!");
			NNAssertEquals(d.get<SerializedArray>("emptyArray").size(), 0, "JSONSerializer failed!");
			NNAssertEquals(d.get<SerializedObject>("emptyObject").size(), 0, "JSONSerializer failed!");
			
			NNAssertEquals(d.get<Serialized *>("nested")->get<bool>("indented"), true, "JSONSerializer failed!");
			NNAssertEquals(d.get<Serialized *>("nested")->get<std::string>("powerLevel"), "> 9000", "JSONSerializer failed!");
		}
		
		// pretty
		{
			std::stringstream ss;
			JSONSerializer::write(s, ss, true);
			
			Serialized d = JSONSerializer::read(ss);
			
			NNAssertEquals(d.get<std::string>("library"), "nnlib", "JSONSerializer failed!");
			NNAssertEquals(d.get<bool>("awesome"), true, "JSONSerializer failed!");
			NNAssertEquals(d.get<bool>("notAwesome"), false, "JSONSerializer failed!");
			NNAssertEquals(d.get<size_t>("number"), 42, "JSONSerializer failed!");
			NNAssertEquals(d.type("nothing"), Serialized::Null, "JSONSerializer failed!");
			NNAssertEquals(d.get<SerializedArray>("emptyArray").size(), 0, "JSONSerializer failed!");
			NNAssertEquals(d.get<SerializedObject>("emptyObject").size(), 0, "JSONSerializer failed!");
			
			NNAssertEquals(d.get<Serialized *>("nested")->get<bool>("indented"), true, "JSONSerializer failed!");
			NNAssertEquals(d.get<Serialized *>("nested")->get<std::string>("powerLevel"), "> 9000", "JSONSerializer failed!");
		}
		
		// from string
		{
			std::string jsonString = "[ \"hello, my name is \\\"bingo\\\"\", 1.22e1, 512e-2 ]";
			std::stringstream ss(jsonString);
			
			Serialized d = JSONSerializer::read(ss);
			
			NNAssertEquals(d.get<std::string>(0), "hello, my name is \"bingo\"", "JSONSerializer failed!");
			NNAssertAlmostEquals(d.get<double>(1), 12.2, 1e-12, "JSONSerializer failed!");
			NNAssertAlmostEquals(d.get<double>(2), 5.12, 1e-12, "JSONSerializer failed!");
			
			std::stringstream ss2;
			JSONSerializer::write(d, ss2);
			d = JSONSerializer::read(ss2);
			
			NNAssertEquals(d.get<std::string>(0), "hello, my name is \"bingo\"", "JSONSerializer failed!");
			NNAssertAlmostEquals(d.get<double>(1), 12.2, 1e-12, "JSONSerializer failed!");
			NNAssertAlmostEquals(d.get<double>(2), 5.12, 1e-12, "JSONSerializer failed!");
		}
		
		// incompatibility
		
		bool ok = false;
		try
		{
			std::stringstream ss("[ this is not json ]");
			JSONSerializer::read(ss);
		}
		catch(const Error &)
		{
			ok = true;
		}
		NNAssert(ok, "JSONSerializer failed! Allowed invalid JSON values!");
	}
	
	// neural network serialization
	
	{
		Sequential<> nn(
			new Linear<>(10, 5),
			new TanH<>(),
			new Linear<>(5, 10),
			new TanH<>()
		);
		
		std::stringstream ss;
		JSONSerializer::write(nn, ss);
		
		Sequential<> *deserialized = JSONSerializer::read(ss).as<Sequential<> *>();
		
		auto &p1 = nn.params();
		auto &p2 = deserialized->params();
		
		for(auto i = p1.begin(), j = p2.begin(), end = p1.end(); i != end; ++i, ++j)
		{
			NNAssertAlmostEquals(*i, *j, 1e-12, "JSONSerializer failed!");
		}
		
		delete deserialized;
	}
}

#endif
