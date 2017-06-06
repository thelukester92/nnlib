#ifndef TEST_ARCHIVE_H
#define TEST_ARCHIVE_H

#include "nnlib/serialization/archive.h"
#include "nnlib/serialization/string.h"
#include "nnlib/nn/module.h"
using namespace nnlib;

template <typename T>
void TestSerializationOfModule(T &module)
{
	OutputStringArchive osa;
	osa(module);
	
	{
		T deserialized;
		
		InputStringArchive isa(osa.str());
		isa(deserialized);
		
		NNAssertEquals(deserialized.inputs(), module.inputs(), "Serialization failed! Mismatching inputs.");
		NNAssertEquals(deserialized.outputs(), module.outputs(), "Serialization failed! Mismatching outputs.");
		
		auto &p1 = module.parameters();
		auto &p2 = deserialized.parameters();
		
		for(auto i = p1.begin(), j = p2.begin(), k = p1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed! Mismatching parameters.");
	}
	
	{
		Module<> *deserialized;
		
		InputStringArchive isa(osa.str());
		isa(deserialized);
		
		NNAssertEquals(deserialized->inputs(), module.inputs(), "Generic serialization failed! Mismatching inputs.");
		NNAssertEquals(deserialized->outputs(), module.outputs(), "Generic serialization failed! Mismatching outputs.");
		
		auto &p1 = module.parameters();
		auto &p2 = deserialized->parameters();
		
		for(auto i = p1.begin(), j = p2.begin(), k = p1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Generic serialization failed! Mismatching parameters.");
		
		delete deserialized;
	}
}

template <typename T>
void TestSerializationOfIterable(T &iterable)
{
	T deserialized;
	
	OutputStringArchive osa;
	osa(iterable);
	
	InputStringArchive isa(osa.str());
	isa(deserialized);
	
	for(auto i = iterable.begin(), j = deserialized.begin(), k = iterable.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed!");
}

void TestArchive()
{
	// todo: fill me in
}

#endif
