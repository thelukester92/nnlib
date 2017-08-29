#ifndef TEST_CSV_SERIALIZER_H
#define TEST_CSV_SERIALIZER_H

#include "nnlib/serialization/csvserializer.h"
#include "nnlib/serialization/serialized.h"

using namespace nnlib;

void TestCSVSerializer()
{
	// basic serialization
	
	{
		Serialized s;
		
		s.add(Serialized::Array);
		s.get<Serialized *>(0)->add(3.14);
		s.get<Serialized *>(0)->add(12);
		s.get<Serialized *>(0)->add("Me");
		
		std::stringstream ss;
		CSVSerializer::write(s, std::cout);
		
		/*
		Serialized d = JSONSerializer::read(ss);
		
		NNAssertEquals(d.get<std::string>("library"), "nnlib", "JSONSerializer failed!");
		NNAssertEquals(d.get<bool>("awesome"), true, "JSONSerializer failed!");
		NNAssertEquals(d.get<size_t>("number"), 42, "JSONSerializer failed!");
		*/
	}
}

#endif
