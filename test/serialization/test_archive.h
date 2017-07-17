#ifndef TEST_ARCHIVE_H
#define TEST_ARCHIVE_H

#include <sstream>
#include "nnlib/serialization/basic.h"
#include "nnlib/tensor.h"
#include "nnlib/nn/linear.h"
#include "nnlib/nn/sequential.h" // sequential uses serialize instead of load/save
using namespace nnlib;

template <typename T, typename InputArchive = BasicInputArchive, typename OutputArchive = BasicOutputArchive>
void TestSerializationOfModule(T &module)
{
	std::stringstream ss1, ss2;
	ss1.precision(16);
	ss2.precision(16);
	
	{
		OutputArchive out(ss1);
		out(module);
		ss2 << ss1.str();
	}
	
	{
		T deserialized;
		
		InputArchive in(ss1);
		in(deserialized);
		deserialized.training(module.training());
		
		NNAssertEquals(deserialized.inputs(), module.inputs(), "Serialization failed! Mismatching inputs.");
		NNAssertEquals(deserialized.outputs(), module.outputs(), "Serialization failed! Mismatching outputs.");
		
		auto &p1 = module.parameters();
		auto &p2 = deserialized.parameters();
		
		for(auto i = p1.begin(), j = p2.begin(), k = p1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed! Mismatching parameters.");
		
		auto tensor = module.output().copy();
		tensor.resize(module.inputs()).rand();
		
		RandomEngine::seed(0);
		auto &o1 = module.forward(tensor);
		
		RandomEngine::seed(0);
		auto &o2 = deserialized.forward(tensor);
		
		for(auto i = o1.begin(), j = o2.begin(), k = o1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed! Different outputs for the same input.");
	}
	
	{
		Module<> *deserialized;
		
		InputArchive in(ss2);
		in(deserialized);
		deserialized->training(module.training());
		
		NNAssertEquals(deserialized->inputs(), module.inputs(), "Generic serialization failed! Mismatching inputs.");
		NNAssertEquals(deserialized->outputs(), module.outputs(), "Generic serialization failed! Mismatching outputs.");
		
		auto &p1 = module.parameters();
		auto &p2 = deserialized->parameters();
		
		for(auto i = p1.begin(), j = p2.begin(), k = p1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Generic serialization failed! Mismatching parameters.");
		
		auto tensor = module.output().copy();
		tensor.resize(module.inputs()).rand();
		
		RandomEngine::seed(0);
		auto &o1 = module.forward(tensor);
		
		RandomEngine::seed(0);
		auto &o2 = deserialized->forward(tensor);
		
		for(auto i = o1.begin(), j = o2.begin(), k = o1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Generic serialization failed! Different outputs for the same input.");
		
		delete deserialized;
	}
}

template <typename T, typename InputArchive = BasicInputArchive, typename OutputArchive = BasicOutputArchive>
void TestSerializationOfIterable(T &iterable)
{
	std::stringstream ss;
	ss.precision(16);
	
	T deserialized;
	
	OutputArchive out(ss);
	out(iterable);
	
	InputArchive in(ss);
	in(deserialized);
	
	for(auto i = iterable.begin(), j = deserialized.begin(), k = iterable.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed!");
}

template <typename InputArchive, typename OutputArchive>
void TestArchive()
{
	Tensor<> t = Tensor<>(3, 2, 4).rand();
	Linear<> l(2, 3, 4);
	Sequential<> s(new Linear<>(2, 3, 4), new Linear<>(3, 2, 4));
	
	TestSerializationOfIterable<Tensor<>, InputArchive, OutputArchive>(t);
	TestSerializationOfModule<Linear<>, InputArchive, OutputArchive>(l);
	TestSerializationOfModule<Sequential<>, InputArchive, OutputArchive>(s);
	
	{
		std::stringstream ss;
		InputArchive in(ss);
		OutputArchive out(ss);
		
		std::string s;
		out("string with spaces");
		in(s);
		NNAssertEquals(s, "string with spaces", "Serialization failed!");
	}
	
	bool ok = false;
	try
	{
		Tensor<> t;
		std::stringstream empty("");
		InputArchive ar(empty);
		ar(t);
	}
	catch(const Error &e)
	{
		ok = true;
	}
	NNAssert(ok, "Intentionally bad deserialization did not throw an error!");
}

#endif
