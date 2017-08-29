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
		s.get<Serialized *>(0)->add("this is a string");
		
		s.add(Serialized::Array);
		s.get<Serialized *>(1)->add("this is a \"string\"");
		s.get<Serialized *>(1)->add("this, is, a, string");
		
		std::stringstream ss;
		CSVSerializer::write(s, ss);
		
		Serialized d = CSVSerializer::read(ss);
		
		NNAssertAlmostEquals(d.get<Serialized *>(0)->get<double>(0), 3.14, 1e-12, "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(0)->get<int>(1), 12, "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(0)->get<std::string>(2), "this is a string", "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(1)->get<std::string>(0), "this is a \"string\"", "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(1)->get<std::string>(1), "this, is, a, string", "CSVSerializer failed!");
	}
}

#endif
