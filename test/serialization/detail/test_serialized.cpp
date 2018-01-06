#include "../test_serialized.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/nn/module.hpp"
#include "nnlib/serialization/serialized.hpp"
#include <vector>
using namespace nnlib;

void TestSerialized()
{
	{
		NNAssertEquals(Serialized(Serialized::Null).type(), Serialized::Null, "Serialized::Serialized(Type) failed!");
		NNAssertEquals(Serialized(Serialized::Boolean).type(), Serialized::Boolean, "Serialized::Serialized(Type) failed!");
		NNAssertEquals(Serialized(Serialized::Integer).type(), Serialized::Integer, "Serialized::Serialized(Type) failed!");
		NNAssertEquals(Serialized(Serialized::Float).type(), Serialized::Float, "Serialized::Serialized(Type) failed!");
		NNAssertEquals(Serialized(Serialized::String).type(), Serialized::String, "Serialized::Serialized(Type) failed!");
		NNAssertEquals(Serialized(Serialized::Array).type(), Serialized::Array, "Serialized::Serialized(Type) failed!");
		NNAssertEquals(Serialized(Serialized::Object).type(), Serialized::Object, "Serialized::Serialized(Type) failed!");
	}
	
	{
		NNAssertEquals(Serialized().type(), Serialized::Null, "Serialized::Serialized) failed!");
		NNAssertEquals(Serialized(nullptr).type(), Serialized::Null, "Serialized::Serialized) failed!");
		NNAssertEquals(Serialized(true).type(), Serialized::Boolean, "Serialized::Serialized(bool) failed!");
		NNAssertEquals(Serialized(42).type(), Serialized::Integer, "Serialized::Serialized(int) failed!");
		NNAssertEquals(Serialized(3.14).type(), Serialized::Float, "Serialized::Serialized(double) failed!");
		NNAssertEquals(Serialized("hello").type(), Serialized::String, "Serialized::Serialized(std::string) failed!");
		
		std::vector<int> v(10);
		v[0] = 5;
		
		Serialized s(v.begin(), v.end());
		NNAssertEquals(s.type(), Serialized::Array, "Serialized::Serialized(iterator, iterator) failed!");
		NNAssertEquals(s.size(), 10, "Serialized::Serialized(iterator, iterator) failed!");
		NNAssertEquals(s.get<int>(0), 5, "Serialized::Serialized(iterator, iterator) failed!");
		
		Serialized t(s);
		NNAssertEquals(t.type(), Serialized::Array, "Serialized::Serialized(iterator, iterator) failed!");
		NNAssertEquals(t.size(), 10, "Serialized::Serialized(iterator, iterator) failed!");
		NNAssertEquals(t.get<int>(0), 5, "Serialized::Serialized(iterator, iterator) failed!");
	}
	
	{
		NNAssertEquals(Serialized(true).get<bool>(), true, "Serialized::as<bool>() failed!");
		NNAssertEquals(Serialized(42).get<int>(), 42, "Serialized::as<int>() failed!");
		NNAssertAlmostEquals(Serialized(3.14).get<double>(), 3.14, 1e-12, "Serialized::as<double>() failed!");
		NNAssertEquals(Serialized("hello").get<std::string>(), "hello", "Serialized::as<std::string>() failed!");
	}
	
	{
		Serialized s;
		s.add(42);
		s.add("hello");
		s.add(Serialized::Array);
		s.add(nullptr);
		
		NNAssertEquals(s.type(0), Serialized::Integer, "Serialized::type(size_t) failed!");
		NNAssertEquals(s.type(1), Serialized::String, "Serialized::type(size_t) failed!");
		NNAssertEquals(s.type(2), Serialized::Array, "Serialized::type(size_t) failed!");
		NNAssertEquals(s.type(3), Serialized::Null, "Serialized::type(size_t) failed!");
		NNAssertEquals(s.get<int>(0), 42, "Serialized::get<int>(int) failed!");
		NNAssertEquals(s.get<std::string>(1), "hello", "Serialized::get<std::string>(int) failed!");
		NNAssertEquals(s.size(2), 0, "Serialized::size(int) failed!");
		NNAssertEquals(s.get(2)->type(), Serialized::Array, "Serialized::get(int) failed!");
		NNAssertEquals(s.get<Tensor<NN_REAL_T> *>(3), nullptr, "Serialized::get<Tensor<NN_REAL_T> *>(int) failed!");
		NNAssertEquals(s.get<Module<NN_REAL_T> *>(3), nullptr, "Serialized::get<Module<NN_REAL_T> *>(int) failed!");
		
		s.set(0, 365);
		NNAssertEquals(s.get<int>(0), 365, "Serialized::set<int>(int) failed!");
		
		s.set(1, new Serialized());
		NNAssertEquals(s.type(1), Serialized::Null, "Serialized::set<Serialized *>(int) failed!");
		
		Module<NN_REAL_T> *m = nullptr;
		s.set(2, m);
		NNAssertEquals(s.type(2), Serialized::Null, "Serialized::set<Module<T> *>(int) failed!");
	}
	
	{
		Serialized t;
		t.add(42);
		t.add("hello");
		t.add(Serialized::Array);
		t.add(nullptr);
		t.add(Tensor<NN_REAL_T>(5).fill(3.14));
		
		const Serialized &s = t;
		
		NNAssertEquals(s.type(0), Serialized::Integer, "Serialized::type(size_t) failed!");
		NNAssertEquals(s.type(1), Serialized::String, "Serialized::type(size_t) failed!");
		NNAssertEquals(s.type(2), Serialized::Array, "Serialized::type(size_t) failed!");
		NNAssertEquals(s.type(3), Serialized::Null, "Serialized::type(size_t) failed!");
		NNAssertEquals(s.get<int>(0), 42, "Serialized::get<int>(int) failed!");
		NNAssertEquals(s.get<std::string>(1), "hello", "Serialized::get<std::string>(int) failed!");
		NNAssertEquals(s.size(2), 0, "Serialized::size(int) failed!");
		NNAssertEquals(s.get<Serialized *>(2)->type(), Serialized::Array, "Serialized::get<Serialized *>(int) failed!");
		NNAssertEquals(s.get<Tensor<NN_REAL_T> *>(3), nullptr, "Serialized::get<Tensor<NN_REAL_T> *>(int) failed!");
		NNAssertEquals(s.get<Tensor<NN_REAL_T> *>(4)->at(0), 3.14, "Serialized::get<Tensor<NN_REAL_T> *>(int) failed!");
		NNAssertEquals(s.get<Module<NN_REAL_T> *>(3), nullptr, "Serialized::get<Module<NN_REAL_T> *>(int) failed!");
	}
	
	{
		Serialized obj;
		obj.set("null", Serialized::Null);
		obj.set("bool", true);
		obj.set("itgr", 42);
		obj.set("flot", 3.14);
		obj.set("stng", "hello");
		obj.set("arry", Serialized::Array);
		obj.set("objt", Serialized::Object);
		obj.set("nptr", nullptr);
		
		NNAssertEquals(obj.get<Serialized *>("null")->size(), 1, "Serialized::size() failed!");
		NNAssertEquals(obj.size(), 8, "Serialized::size() failed!");
		NNAssert(obj.has("null"), "Serialized::has(string) failed!");
		NNAssert(!obj.has("bull"), "Serialized::has(string) failed!");
		
		NNAssertEquals(obj.get("null")->convertTo<bool>(), false, "Serialized::convertTo<bool>() failed for a null value!");
		NNAssertEquals(obj.get("bool")->convertTo<bool>(), true, "Serialized::convertTo<bool>() failed for a bool value!");
		NNAssertEquals(obj.get("itgr")->convertTo<bool>(), true, "Serialized::convertTo<bool>() failed for an integer value!");
		NNAssertEquals(obj.get("flot")->convertTo<bool>(), true, "Serialized::convertTo<bool>() failed for a floating point value!");
		
		bool ok = false;
		try
		{
			obj.get("stng")->convertTo<bool>();
		}
		catch(const Error &)
		{
			ok = true;
		}
		NNAssert(ok, "Serialized::convertTo<bool> did not throw an error for an unexpected type!");
		
		NNAssertEquals(obj.get("null")->convertTo<int>(), 0, "Serialized::convertTo<int>() failed for a null value!");
		NNAssertEquals(obj.get("bool")->convertTo<int>(), 1, "Serialized::convertTo<int>() failed for a bool value!");
		NNAssertEquals(obj.get("itgr")->convertTo<int>(), 42, "Serialized::convertTo<int>() failed for an integer value!");
		NNAssertEquals(obj.get("flot")->convertTo<int>(), 3, "Serialized::convertTo<int>() failed for a floating point value!");
		
		ok = false;
		try
		{
			obj.get("stng")->convertTo<int>();
		}
		catch(const Error &)
		{
			ok = true;
		}
		NNAssert(ok, "Serialized::convertTo<int> did not throw an error for an unexpected type!");
		
		NNAssertEquals(obj.get("null")->convertTo<double>(), 0, "Serialized::convertTo<double>() failed for a null value!");
		NNAssertEquals(obj.get("bool")->convertTo<double>(), 1, "Serialized::convertTo<double>() failed for a bool value!");
		NNAssertEquals(obj.get("itgr")->convertTo<double>(), 42, "Serialized::convertTo<double>() failed for an integer value!");
		NNAssertAlmostEquals(obj.get("flot")->convertTo<double>(), 3.14, 1e-12, "Serialized::convertTo<double>() failed for a floating point value!");
		
		ok = false;
		try
		{
			obj.get("stng")->convertTo<double>();
		}
		catch(const Error &)
		{
			ok = true;
		}
		NNAssert(ok, "Serialized::convertTo<double> did not throw an error for an unexpected type!");
		
		NNAssertEquals(obj.get("null")->convertTo<std::string>(), "null", "Serialized::convertTo<string>() failed for a null value!");
		NNAssertEquals(obj.get("bool")->convertTo<std::string>(), "true", "Serialized::convertTo<string>() failed for a bool value!");
		NNAssertEquals(obj.get("itgr")->convertTo<std::string>(), "42", "Serialized::convertTo<string>() failed for an integer value!");
		NNAssertAlmostEquals(std::atof(obj.get("flot")->convertTo<std::string>().c_str()), 3.14, 1e-12, "Serialized::convertTo<string>() failed for a floating point value!");
		NNAssertEquals(obj.get("stng")->convertTo<std::string>(), "hello", "Serialized::convertTo<string>() failed for a string value!");
		
		ok = false;
		try
		{
			obj.get("arry")->convertTo<std::string>();
		}
		catch(const Error &)
		{
			ok = true;
		}
		NNAssert(ok, "Serialized::convertTo<string> did not throw an error for an unexpected type!");
		
		Serialized s;
		
		s = *obj.get<Serialized *>("null");
		NNAssertEquals(s.type(), Serialized::Null, "Serialized::operator=(const Serialized &other) failed!");
		
		s = *obj.get<Serialized *>("bool");
		NNAssertEquals(s.type(), Serialized::Boolean, "Serialized::operator=(const Serialized &other) failed!");
		
		s = *obj.get<Serialized *>("itgr");
		NNAssertEquals(s.type(), Serialized::Integer, "Serialized::operator=(const Serialized &other) failed!");
		
		s = *obj.get<Serialized *>("flot");
		NNAssertEquals(s.type(), Serialized::Float, "Serialized::operator=(const Serialized &other) failed!");
		
		s = *obj.get<Serialized *>("stng");
		NNAssertEquals(s.type(), Serialized::String, "Serialized::operator=(const Serialized &other) failed!");
		
		s = *obj.get<Serialized *>("objt");
		NNAssertEquals(s.type(), Serialized::Object, "Serialized::operator=(const Serialized &other) failed!");
		
		s = *obj.get<Serialized *>("nptr");
		NNAssertEquals(s.type(), Serialized::Null, "Serialized::operator=(const Serialzied &other) failed!");
		
		s = std::move(*obj.get<Serialized *>("null"));
		NNAssertEquals(s.type(), Serialized::Null, "Serialized::operator=(Serialized &&other) failed!");
		
		s = std::move(*obj.get<Serialized *>("bool"));
		NNAssertEquals(s.type(), Serialized::Boolean, "Serialized::operator=(Serialized &&other) failed!");
		
		s = std::move(*obj.get<Serialized *>("itgr"));
		NNAssertEquals(s.type(), Serialized::Integer, "Serialized::operator=(Serialized &&other) failed!");
		
		s = std::move(*obj.get<Serialized *>("flot"));
		NNAssertEquals(s.type(), Serialized::Float, "Serialized::operator=(Serialized &&other) failed!");
		
		s = std::move(*obj.get<Serialized *>("stng"));
		NNAssertEquals(s.type(), Serialized::String, "Serialized::operator=(Serialized &&other) failed!");
		
		s = std::move(*obj.get<Serialized *>("objt"));
		NNAssertEquals(s.type(), Serialized::Object, "Serialized::operator=(Serialized &&other) failed!");
		
		s = std::move(*obj.get<Serialized *>("nptr"));
		NNAssertEquals(s.type(), Serialized::Null, "Serialized::operator=(Serialized &&other) failed!");
	}
}
