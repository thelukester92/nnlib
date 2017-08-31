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
		s.get<Serialized *>(0)->add(-12);
		s.get<Serialized *>(0)->add(true);
		s.get<Serialized *>(0)->add(false);
		s.get<Serialized *>(0)->add(Serialized::Null);
		s.get<Serialized *>(0)->add("this is a string");
		
		s.add(Serialized::Array);
		s.get<Serialized *>(1)->add("this is a \"string\"");
		s.get<Serialized *>(1)->add("this, is, a, string");
		s.get<Serialized *>(1)->add("123.456.789");
		
		std::stringstream ss;
		CSVSerializer::write(s, ss);
		
		Serialized d = CSVSerializer::read(ss);
		
		NNAssertAlmostEquals(d.get<Serialized *>(0)->get<double>(0), 3.14, 1e-12, "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(0)->get<int>(1), -12, "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(0)->get<std::string>(2), "true", "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(0)->get<std::string>(3), "false", "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(0)->get<std::string>(4), "null", "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(0)->get<std::string>(5), "this is a string", "CSVSerializer failed!");
		
		NNAssertEquals(d.get<Serialized *>(1)->get<std::string>(0), "this is a \"string\"", "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(1)->get<std::string>(1), "this, is, a, string", "CSVSerializer failed!");
		NNAssertEquals(d.get<Serialized *>(1)->get<std::string>(2), "123.456.789", "CSVSerializer failed!");
		
		bool ok = false;
		Serialized notCompatibleWithCSV(Serialized::Object);
		try
		{
			CSVSerializer::write(notCompatibleWithCSV, ss);
		}
		catch(const Error &)
		{
			ok = true;
		}
		NNAssert(ok, "CSVSerializer failed! Accepted an object instead of an array!");
		
		ok = false;
		notCompatibleWithCSV.add(Serialized::Object);
		try
		{
			CSVSerializer::write(notCompatibleWithCSV, ss);
		}
		catch(const Error &)
		{
			ok = true;
		}
		NNAssert(ok, "CSVSerializer failed! Accepted an object instead of an array!");
		
		ok = false;
		notCompatibleWithCSV.get<Serialized *>(0)->add(Serialized::Object);
		try
		{
			CSVSerializer::write(notCompatibleWithCSV, ss);
		}
		catch(const Error &)
		{
			ok = true;
		}
		NNAssert(ok, "CSVSerializer failed! Accepted an object instead of a number or string!");
	}
}

#endif
