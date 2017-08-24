#ifndef TEST_STORAGE_H
#define TEST_STORAGE_H

#include "nnlib/core/error.h"
#include "nnlib/core/storage.h"
using namespace nnlib;

void TestStorage()
{
	// test constructors
	
	Storage<double> empty;
	NNAssertEquals(empty.size(), 0, "Storage::Storage() failed!");
	
	Storage<double> regular(5, 3.14);
	NNAssertEquals(regular.size(), 5, "Storage::Storage(size_t, T) failed! Wrong size!");
	NNAssertEquals(*regular.ptr(), 3.14, "Storage::Storage(size_t, T) failed! Wrong value!");
	
	Storage<double> copy(regular);
	NNAssertEquals(copy.size(), 5, "Storage::Storage(Storage) failed! Wrong size!");
	NNAssertEquals(*copy.ptr(), 3.14, "Storage::Storage(Storage) failed! Wrong value!");
	
	Storage<double> initialized({ 1.0, 2.0, 3.0, 4.0 });
	NNAssertEquals(initialized.size(), 4, "Storage::Storage(initializer_list) failed! Wrong size!");
	NNAssertEquals(*initialized.ptr(), 1.0, "Storage::Storage(initializer_list) failed! Wrong value!");
	
	// test equality checks
	
	NNAssertEquals(empty, Storage<double>(), "Storage::operator== failed!");
	NNAssertEquals(regular, Storage<double>({ 3.14, 3.14, 3.14, 3.14, 3.14 }), "Storage::operator== failed!");
	NNAssertEquals(copy, Storage<double>({ 3.14, 3.14, 3.14, 3.14, 3.14 }), "Storage::operator== failed!");
	NNAssertEquals(initialized, Storage<double>({ 1.0, 2.0, 3.0, 4.0 }), "Storage::operator== failed!");
	NNAssertEquals(regular, copy, "Storage::operator== failed!");
	
	NNAssertNotEquals(empty, regular, "Storage::operator!=(Storage) failed!");
	NNAssertNotEquals(regular, initialized, "Storage::operator!=(Storage) failed!");
	NNAssertNotEquals(initialized, Storage<double>({ 2.0 }), "Storage::operator!=(initializer_list) failed!");
	
	// test assignment
	
	empty = initialized;
	NNAssertEquals(empty, initialized, "Storage::operator=(Storage) failed!");
	
	empty = {};
	NNAssertEquals(empty, Storage<double>(), "Storage::operator=(initializer_list) failed!");
	
	// test other methods
	
	regular.resize(7, 6.28);
	NNAssertEquals(regular, Storage<double>({ 3.14, 3.14, 3.14, 3.14, 3.14, 6.28, 6.28 }), "Storage::resize failed!");
	
	regular.push_back(31.4);
	NNAssertEquals(regular, Storage<double>({ 3.14, 3.14, 3.14, 3.14, 3.14, 6.28, 6.28, 31.4 }), "Storage::push_back failed!");
	
	regular.erase(0);
	NNAssertEquals(regular, Storage<double>({ 3.14, 3.14, 3.14, 3.14, 6.28, 6.28, 31.4 }), "Storage::erase(0) failed!");
	
	regular.erase(4);
	NNAssertEquals(regular, Storage<double>({ 3.14, 3.14, 3.14, 3.14, 6.28, 31.4 }), "Storage::erase(4) failed!");
	
	regular.erase(regular.size() - 1);
	NNAssertEquals(regular, Storage<double>({ 3.14, 3.14, 3.14, 3.14, 6.28 }), "Storage::erase(size() - 1) failed!");
	
	regular.clear();
	NNAssertEquals(regular, Storage<double>(), "Storage::clear failed!");
	
	NNAssertEquals(copy[0], 3.14, "Storage::operator[] failed!");
	NNAssertEquals(initialized.front(), 1.0, "Storage::front failed!");
	NNAssertEquals(initialized.back(), 4.0, "Storage::back failed!");
	
	size_t i = 0;
	for(auto &v : copy)
	{
		NNAssertEquals(&v, &copy[i], "Storage::begin and/or Storage::end failed!");
		++i;
	}
	
	// test const methods
	
	const Storage<double> &constant = copy;
	NNAssertEquals((size_t)(constant.end() - constant.begin()), constant.size(), "const Storage::begin and/or const Storage::end failed!");
	NNAssertEquals(constant.ptr(), copy.ptr(), "const Storage::ptr failed!");
	NNAssertEquals(constant[2], copy[2], "const Storage::operator[] failed!");
	NNAssertEquals(constant.front(), copy.front(), "const Storage::front failed!");
	NNAssertEquals(constant.back(), copy.back(), "const Storage::back failed!");
	
	// test serialization
	
	Storage<size_t> serializable = { 1, 2, 3, 4, 5 };
	Storage<size_t> serialized;
	
	SerializedNode node;
	serializable.save(node);
	serialized.load(node);
	NNAssertEquals(serializable, serialized, "Storage::save and/or Storage::load failed!");
}

#endif
