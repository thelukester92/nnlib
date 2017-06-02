#ifndef TEST_TENSOR_H
#define TEST_TENSOR_H

#include "nnlib/tensor.h"
using namespace nnlib;

void TestTensor()
{
	// test constructors
	
	Tensor<> empty;
	NNAssertEquals(empty.size(), 0, "Tensor::Tensor() failed!");
	
	Tensor<> vector(3, 4);
	NNAssertEquals(vector.dims(), 2, "Tensor::Tensor(size_t, size_t) failed! Wrong dimensionality!");
	NNAssertEquals(vector.size(), 12, "Tensor::Tensor(size_t, size_t) failed! Wrong size!");
	NNAssertEquals(*vector.ptr(), 0.0, "Tensor::Tensor(size_t, size_t) failed! Wrong value!");
	
	Tensor<> initFromStorage(Storage<double>({ 3.14, 42.0 }));
	NNAssertEquals(initFromStorage.dims(), 1, "Tensor::Tensor(Storage) failed! Wrong dimensionality!");
	NNAssertEquals(initFromStorage.size(), 2, "Tensor::Tensor(Storage) failed! Wrong size!");
	NNAssertEquals(*initFromStorage.ptr(), 3.14, "Tensor::Tensor(Storage) failed! Wrong value!");
	
	Tensor<> initFromList({ 1.0, 2.0, 3.0, 4.0 });
	NNAssertEquals(initFromList.dims(), 1, "Tensor::Tensor(initializer_list) failed! Wrong dimensionality!");
	NNAssertEquals(initFromList.size(), 4, "Tensor::Tensor(initializer_list) failed! Wrong size!");
	NNAssertEquals(*initFromList.ptr(), 1.0, "Tensor::Tensor(initializer_list) failed! Wrong value!");
	
	Tensor<> initWithDims({ 4, 2, 3 }, true);
	NNAssertEquals(initWithDims.dims(), 3, "Tensor::Tensor(Storage, bool) failed! Wrong dimensionality!");
	NNAssertEquals(initWithDims.size(), 24, "Tensor::Tensor(Storage, bool) failed! Wrong size!");
	
	Tensor<> view(vector);
	NNAssertEquals(view.shape(), vector.shape(), "Tensor::Tensor(Tensor &) failed! Wrong shape!");
	NNAssertEquals(view.ptr(), vector.ptr(), "Tensor::Tensor(Tensor &) failed! Wrong data!");
	
	Tensor<> viewOfMoved(std::move(initWithDims));
	NNAssertEquals(viewOfMoved.shape(), initWithDims.shape(), "Tensor::Tensor(Tensor &&) failed! Wrong shape!");
	NNAssertEquals(viewOfMoved.ptr(), initWithDims.ptr(), "Tensor::Tensor(Tensor &&) failed! Wrong data!");
	
	// test assignment
	
	empty = Storage<double>({ 1.0, 2.0, 3.0 });
	NNAssertEquals(empty.dims(), 1, "Tensor::operator=(Storage) failed! Wrong dimensionality!");
	NNAssertEquals(empty.size(), 3, "Tensor::operator=(Storage) failed! Wrong shape!");
	NNAssertEquals(*empty.ptr(), 1.0, "Tensor::operator=(Storage) failed! Wrong data!");
	
	empty = { 3.0, 6.0, 9.0, 12.0 };
	NNAssertEquals(empty.dims(), 1, "Tensor::operator=(initializer_list) failed! Wrong dimensionality!");
	NNAssertEquals(empty.size(), 4, "Tensor::operator=(initializer_list) failed! Wrong shape!");
	NNAssertEquals(*empty.ptr(), 3.0, "Tensor::operator=(Storage) failed! Wrong data!");
	
	empty = vector;
	NNAssertEquals(empty.shape(), vector.shape(), "Tensor::operator=(Tensor &) failed! Wrong shape!");
	NNAssertEquals(empty.ptr(), vector.ptr(), "Tensor::operator=(Tensor &) failed! Wrong data!");
	
	empty = std::move(initFromStorage);
	NNAssertEquals(empty.shape(), initFromStorage.shape(), "Tensor::operator=(Tensor &&) failed! Wrong shape!");
	NNAssertEquals(empty.ptr(), initFromStorage.ptr(), "Tensor::operator=(Tensor &&) failed! Wrong data!");
	
	// test element access
	
	vector(2, 2) = 3.14;
	NNAssertEquals(vector(2, 2), 3.14, "Tensor::operator() failed!");
	NNAssertEquals(view(2, 2), 3.14, "Tensor::operator() failed!");
	NNAssertEquals(&vector(2, 2), &view(2, 2), "Tensor::operator() failed!");
	
	vector.fill(6.26);
	for(auto &v : vector)
		NNAssertEquals(v, 6.26, "Tensor::fill failed!");
	
	// test other methods
	
	empty.resize(11, 22, 33);
	NNAssertEquals(empty.dims(), 3, "Tensor::resize(size_t, size_t, size_t) failed! Wrong dimensionality!");
	NNAssertEquals(empty.size(0), 11, "Tensor::resize(size_t, size_t, size_t) failed!");
	NNAssertEquals(empty.size(1), 22, "Tensor::resize(size_t, size_t, size_t) failed!");
	NNAssertEquals(empty.size(2), 33, "Tensor::resize(size_t, size_t, size_t) failed!");
	NNAssertEquals(empty.size(), 11 * 22 * 33, "Tensor::resize(size_t, size_t, size_t) failed!");
	
	empty.resize(Storage<size_t>({ 2, 4, 6 }));
	NNAssertEquals(empty.shape(), Storage<size_t>({ 2, 4, 6 }), "Tensor::resize(Storage) failed!");
	
	empty.resizeDim(1, 18);
	NNAssertEquals(empty.size(0), 2, "Tensor::resizeDim failed!");
	NNAssertEquals(empty.size(1), 18, "Tensor::resizeDim failed!");
	NNAssertEquals(empty.size(2), 6, "Tensor::resizeDim failed!");
	NNAssertEquals(empty.size(), 2 * 18 * 6, "Tensor::resizeDim failed!");
	
	view = empty.view(2, 2, 2, 2);
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 2, 2, 2 }), "Tensor::view failed! Wrong shape!");
	NNAssertEquals(view.ptr(), empty.ptr(), "Tensor::view failed! Wrong data!");
	
	view = initWithDims.view(Storage<size_t>({ 3, 6 }));
	NNAssertEquals(view.shape(), Storage<size_t>({ 3, 6 }), "Tensor::view failed! Wrong shape!");
	NNAssertEquals(view.ptr(), initWithDims.ptr(), "Tensor::view failed! Wrong data!");
	
	view = empty.reshape(Storage<size_t>({ 2, 2, 9, 2, 3 }));
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 2, 9, 2, 3 }), "Tensor::reshape failed! Wrong shape!");
	NNAssertNotEquals(view.ptr(), empty.ptr(), "Tensor::reshape failed! Wrong data!");
	
	view = empty.reshape(18, 12);
	NNAssertEquals(view.shape(), Storage<size_t>({ 18, 12 }), "Tensor::reshape failed! Wrong shape!");
	NNAssertNotEquals(view.ptr(), empty.ptr(), "Tensor::reshape failed! Wrong data!");
	
	view = empty.select(1, 16);
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 6 }), "Tensor::select failed! Wrong shape!");
	NNAssertEquals(&view(1, 4), &empty(1, 16, 4), "Tensor::select failed! Wrong data!");
	
	view = empty.narrow(2, 3, 3);
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 18, 3 }), "Tensor::narrow failed! Wrong shape!");
	NNAssertEquals(&view(1, 2, 1), &empty(1, 2, 4), "Tensor::narrow failed! Wrong data!");
	
	empty.resize(2, 1, 4);
	view = empty.expand(1, 3);
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 3, 4 }), "Tensor::expand failed! Wrong shape!");
	NNAssertEquals(&view(1, 0, 2), &view(1, 1, 2), "Tensor::expand failed! Wrong data!");
	
	vector.resize(6, 9);
	vector.sub(view, { { 2, 3 }, { 3, 3 } });
	NNAssertEquals(&view.storage(), &vector.storage(), "Tensor::sub(Tensor, initializer_list) failed! Wrong data!");
	NNAssertEquals(&view(1, 1), &vector(3, 4), "Tensor::sub(Tensor, initializer_list) failed! Wrong data!");
	
	viewOfMoved = vector.sub({ { 1, 3 }, { 4, 3 } });
	NNAssertEquals(&viewOfMoved.storage(), &vector.storage(), "Tensor::sub(initializer_list) failed! Wrong data!");
	NNAssertEquals(&viewOfMoved(1, 1), &vector(2, 5), "Tensor::sub(initializer_list) failed! Wrong data!");
	NNAssertEquals(&viewOfMoved(1, 1), &view(0, 2), "Tensor::sub(initializer_list) failed! Wrong data!");
	
	empty = view.copy();
	NNAssertEquals(empty.shape(), view.shape(), "Tensor::copy() failed! Wrong shape!");
	NNAssertNotEquals(&empty.storage(), &view.storage(), "Tensor::copy() failed! Wrong data!");
	
	empty.copy(viewOfMoved);
	NNAssertEquals(empty.shape(), viewOfMoved.shape(), "Tensor::copy(Tensor &) failed! Wrong shape!");
	for(auto x = empty.begin(), y = viewOfMoved.begin(); x != empty.end(); ++x, ++y)
	{
		NNAssertEquals(*x, *y, "Tensor::copy(Tensor &) failed! Wrong data!");
		NNAssertNotEquals(&*x, &*y, "Tensor::copy(Tensor &) failed! Wrong data!");
	}
	
	empty = Tensor<>(3, 4).fill(1.0);
	view = Tensor<>(3, 4).fill(2.0);
	
	empty.swap(view);
	for(auto x = empty.begin(), y = view.begin(); x != empty.end(); ++x, ++y)
	{
		NNAssertEquals(*x, 2.0, "Tensor::swap(Tensor &) failed!");
		NNAssertEquals(*y, 1.0, "Tensor::swap(Tensor &) failed!");
	}
	
	empty.swap(std::move(view));
	for(auto x = empty.begin(), y = view.begin(); x != empty.end(); ++x, ++y)
	{
		NNAssertEquals(*x, 1.0, "Tensor::swap(Tensor &&) failed!");
		NNAssertEquals(*y, 2.0, "Tensor::swap(Tensor &&) failed!");
	}
	
	vector.resize(3, 2);
	vector.select(1, 0).fill(1.0);
	vector.select(1, 1).fill(2.0);
	vector = vector.transpose();
	for(auto &v : vector.select(0, 0))
		NNAssertEquals(v, 1.0, "Tensor::transpose failed!");
	for(auto &v : vector.select(0, 1))
		NNAssertEquals(v, 2.0, "Tensor::transpose failed!");
	
	vector.zeros();
	for(auto &v : vector)
		NNAssertEquals(v, 0.0, "Tensor::zeros failed!");
	
	vector.ones();
	for(auto &v : vector)
		NNAssertEquals(v, 1.0, "Tensor::zeros failed!");
	
	vector.rand(5, 10);
	for(auto &v : vector)
	{
		NNAssertGreaterThanOrEquals(v, 5, "Tensor::rand failed!");
		NNAssertLessThanOrEquals(v, 10, "Tensor::rand failed!");
	}
	
	vector.randn(700, 1);
	for(auto &v : vector)
		NNAssertAlmostEquals(v, 700, 100, "Tensor::randn(T, T) failed! It produced very distant outliers.");
	
	vector.randn(5280, 2, 5);
	for(auto &v : vector)
		NNAssertAlmostEquals(v, 5280, 5, "Tensor::randn(T, T, T) failed!");
	
	view = vector.copy().scale(2);
	for(auto x = view.begin(), y = vector.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, 2 * *y, 1e-12, "Tensor::scale failed!");
	
	view = vector.copy().add(2);
	for(auto x = view.begin(), y = vector.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, 2 + *y, 1e-12, "Tensor::add(T) failed!");
	
	view.resize(12).rand();
	vector.resize(12).rand();
	empty = view.copy().addV(vector);
	for(auto x = view.begin(), y = vector.begin(), z = empty.begin(); x != view.end(); ++x, ++y, ++z)
		NNAssertAlmostEquals(*x + *y, *z, 1e-12, "Tensor::addV failed!");
	
	// test tensor math
	
	view = Tensor<>(3, 100).rand();
	vector = Tensor<>(100).rand();
	viewOfMoved = Tensor<>(view.size(0));
	empty = Tensor<>(view.size(0));
	
	empty.assignMV(view, vector);
	for(int i = 0; i < view.size(0); ++i)
		for(int j = 0; j < view.size(1); ++j)
			viewOfMoved(i) += view(i, j) * vector(j);
	
	for(auto x = viewOfMoved.begin(), y = empty.begin(); x != viewOfMoved.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::assignMV failed!");
	
	vector.resize(3);
	viewOfMoved.resize(100).zeros();
	empty.resize(100);
	
	empty.assignMTV(view, vector);
	for(int i = 0; i < view.size(1); ++i)
		for(int j = 0; j < view.size(0); ++j)
			viewOfMoved(i) += view(j, i) * vector(j);
	
	for(auto x = viewOfMoved.begin(), y = empty.begin(); x != viewOfMoved.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::assignMTV failed!");
	
	// test const methods
	/*
	
	view(Storage)
	view(...)
	select()
	narrow()
	expand()
	sub()
	
	// test Serialization
	
	Storage<double> *deserialized = nullptr;
	Archive::fromString((Archive::toString() << regular).str()) >> deserialized;
	NNAssertNotEquals(deserialized, nullptr, "Storage::save and/or Storage::load failed!");
	for(auto x = deserialized->begin(), y = regular.begin(); x != deserialized->end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Storage::save and/or Storage::load failed!");
	delete deserialized;
	
	*/
}

#endif
