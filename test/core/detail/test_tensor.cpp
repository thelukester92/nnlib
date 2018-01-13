#include "../test_tensor.hpp"
#include "nnlib/core/tensor.hpp"
using namespace nnlib;

void TestTensor()
{
    // test empty concat

    {
        Tensor<NN_REAL_T> empty = Tensor<NN_REAL_T>::concatenate({});
        NNAssertEquals(empty.size(), 0, "Tensor::concatenate on an empty list failed!");
    }

    // test resizing to the same size

    {
        Tensor<NN_REAL_T> t(3);
        t.resizeDim(0, 3);
        NNAssertEquals(t.size(0), 3, "Tensor::resizeDim to the same size failed!");
    }

    // test constructors

    Tensor<NN_REAL_T> empty;
    NNAssertEquals(empty.size(), 0, "Tensor::Tensor() failed!");

    Tensor<NN_REAL_T> vector(3, 4);
    NNAssertEquals(vector.dims(), 2, "Tensor::Tensor(size_t, size_t) failed! Wrong dimensionality!");
    NNAssertEquals(vector.size(), 12, "Tensor::Tensor(size_t, size_t) failed! Wrong size!");
    NNAssertEquals(*vector.ptr(), 0.0, "Tensor::Tensor(size_t, size_t) failed! Wrong value!");

    Tensor<NN_REAL_T> initFromStorage(Storage<double>({ 3.14, 42.0 }));
    NNAssertEquals(initFromStorage.dims(), 1, "Tensor::Tensor(Storage) failed! Wrong dimensionality!");
    NNAssertEquals(initFromStorage.size(), 2, "Tensor::Tensor(Storage) failed! Wrong size!");
    NNAssertEquals(*initFromStorage.ptr(), 3.14, "Tensor::Tensor(Storage) failed! Wrong value!");

    Tensor<NN_REAL_T> initFromList({ 1.0, 2.0, 3.0, 4.0 });
    NNAssertEquals(initFromList.dims(), 1, "Tensor::Tensor(initializer_list) failed! Wrong dimensionality!");
    NNAssertEquals(initFromList.size(), 4, "Tensor::Tensor(initializer_list) failed! Wrong size!");
    NNAssertEquals(*initFromList.ptr(), 1.0, "Tensor::Tensor(initializer_list) failed! Wrong value!");

    Tensor<NN_REAL_T> initWithDims({ 4, 2, 3 }, true);
    NNAssertEquals(initWithDims.dims(), 3, "Tensor::Tensor(Storage, bool) failed! Wrong dimensionality!");
    NNAssertEquals(initWithDims.size(), 24, "Tensor::Tensor(Storage, bool) failed! Wrong size!");

    Tensor<NN_REAL_T> view(vector);
    NNAssertEquals(view.shape(), vector.shape(), "Tensor::Tensor(Tensor &) failed! Wrong shape!");
    NNAssertEquals(view.ptr(), vector.ptr(), "Tensor::Tensor(Tensor &) failed! Wrong data!");

    Tensor<NN_REAL_T> viewOfMoved(std::move(initWithDims));
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
    NNAssertGreaterThan(empty.sharedCount(), 1, "Tensor::operator=(Tensor &) failed! Not sharing data!");
    NNAssert(empty.sharedWith(vector), "Tensor::operator=(Tensor &) failed! Not sharing data!");

    empty = std::move(initFromStorage);
    NNAssertEquals(empty.shape(), initFromStorage.shape(), "Tensor::operator=(Tensor &&) failed! Wrong shape!");
    NNAssertEquals(empty.ptr(), initFromStorage.ptr(), "Tensor::operator=(Tensor &&) failed! Wrong data!");

    empty = Tensor<NN_REAL_T>(3, 3).rand().transpose();
    NNAssert(!empty.contiguous(), "Tensor::transpose failed to make discontiguous!");

    empty.makeContiguous();
    NNAssert(empty.contiguous(), "Tensor::makeContiguous failed!");

    // test element access

    vector(2, 2) = 3.14;
    NNAssertEquals(vector(2, 2), 3.14, "Tensor::operator() failed!");
    NNAssertEquals(view(2, 2), 3.14, "Tensor::operator() failed!");
    NNAssertEquals(&vector(2, 2), &view(2, 2), "Tensor::operator() failed!");
    NNAssertEquals(vector.at(2, 2), view.at(2, 2), "Tensor::at(size_t) failed!");

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
    NNAssertEquals(&view.data(), &vector.data(), "Tensor::sub(Tensor, initializer_list) failed! Wrong data!");
    NNAssertEquals(&view(1, 1), &vector(3, 4), "Tensor::sub(Tensor, initializer_list) failed! Wrong data!");

    viewOfMoved = vector.sub({ { 1, 3 }, { 4, 3 } });
    NNAssertEquals(&viewOfMoved.data(), &vector.data(), "Tensor::sub(initializer_list) failed! Wrong data!");
    NNAssertEquals(&viewOfMoved(1, 1), &vector(2, 5), "Tensor::sub(initializer_list) failed! Wrong data!");
    NNAssertEquals(&viewOfMoved(1, 1), &view(0, 2), "Tensor::sub(initializer_list) failed! Wrong data!");

    empty = view.copy();
    NNAssertEquals(empty.shape(), view.shape(), "Tensor::copy() failed! Wrong shape!");
    NNAssertNotEquals(&empty.data(), &view.data(), "Tensor::copy() failed! Wrong data!");

    empty.copy(viewOfMoved);
    NNAssertEquals(empty.shape(), viewOfMoved.shape(), "Tensor::copy(Tensor &) failed! Wrong shape!");
    for(auto x = empty.begin(), y = viewOfMoved.begin(); x != empty.end(); ++x, ++y)
    {
        NNAssertEquals(*x, *y, "Tensor::copy(Tensor &) failed! Wrong data!");
        NNAssertNotEquals(&*x, &*y, "Tensor::copy(Tensor &) failed! Wrong data!");
    }

    empty = Tensor<NN_REAL_T>(3, 4).fill(1.0);
    view = Tensor<NN_REAL_T>(3, 4).fill(2.0);

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

    NNAssertNotEquals(vector.begin(), view.begin(), "TensorIterator::operator== failed!");

    view = Tensor<NN_REAL_T>({
        1, 4, 7,
        2, 5, 8,
        3, 6, 9
    }).resize(3, 3).transpose();

    {
        auto itr = view.begin();
        NNAssertAlmostEquals(*itr++, 1, 1e-12, "TensorIterator failed for discontiguous tensor!");
        NNAssertAlmostEquals(*itr++, 2, 1e-12, "TensorIterator failed for discontiguous tensor!");
        NNAssertAlmostEquals(*itr++, 3, 1e-12, "TensorIterator failed for discontiguous tensor!");
        NNAssertAlmostEquals(*itr++, 4, 1e-12, "TensorIterator failed for discontiguous tensor!");
        NNAssertAlmostEquals(*itr++, 5, 1e-12, "TensorIterator failed for discontiguous tensor!");
        NNAssertAlmostEquals(*itr++, 6, 1e-12, "TensorIterator failed for discontiguous tensor!");
        NNAssertAlmostEquals(*itr++, 7, 1e-12, "TensorIterator failed for discontiguous tensor!");
        NNAssertAlmostEquals(*itr++, 8, 1e-12, "TensorIterator failed for discontiguous tensor!");
        NNAssertAlmostEquals(*itr++, 9, 1e-12, "TensorIterator failed for discontiguous tensor!");
    }

    // test sparsification / unsparsification

    view = Tensor<NN_REAL_T>({
        3, 3, 0,
        0, 0, 1,
        1, 1, 1,
        2, 2, 1
    }).resize(4, 3);
    empty = Tensor<NN_REAL_T>({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    }).resize(3, 3).sparsify();

    forEach([&](NN_REAL_T x, NN_REAL_T y)
    {
        NNAssertAlmostEquals(x, y, 1e-12, "Tensor::sparsify failed!");
    }, view, empty);

    view = Tensor<NN_REAL_T>({
        3, 3, 0,
        0, 0, 1,
        1, 1, 1,
        2, 2, 1
    }).resize(4, 3).unsparsify();
    empty = Tensor<NN_REAL_T>({
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    }).resize(3, 3);

    forEach([&](NN_REAL_T x, NN_REAL_T y)
    {
        NNAssertAlmostEquals(x, y, 1e-12, "Tensor::unsparsify failed!");
    }, view, empty);

    // test const methods

    view.resize(10, 10);

    const Tensor<NN_REAL_T> &constant = view;
    NNAssertEquals(&constant.at(0, 0), &view.at(0, 0), "const Tensor::at(size_t) failed!");
    NNAssertEquals(&constant.data(), &view.data(), "const Tensor::data(size_t) failed!");

    {
        const Tensor<NN_REAL_T> &constView = constant.view(Storage<size_t>({ 3, 3 }));
        NNAssertEquals(constView.shape(), Storage<size_t>({ 3, 3 }), "const Tensor::view(Storage) failed! Wrong shape!");
        NNAssertEquals(&constView(0, 0), &constant(0, 0), "const Tensor::view(Storage) failed! Wrong data!");
    }

    {
        const Tensor<NN_REAL_T> &constView = constant.view(3, 3);
        NNAssertEquals(constView.shape(), Storage<size_t>({ 3, 3 }), "const Tensor::view(size_t, size_t) failed! Wrong shape!");
        NNAssertEquals(&constView(0, 0), &constant(0, 0), "const Tensor::view(size_t, size_t) failed! Wrong data!");
    }

    {
        const Tensor<NN_REAL_T> &constView = constant.select(1, 1);
        NNAssertEquals(constView.shape(), Storage<size_t>({ 10 }), "const Tensor::select failed! Wrong shape!");
        NNAssertEquals(&constView(2), &constant(2, 1), "const Tensor::select failed! Wrong data!");
    }

    {
        const Tensor<NN_REAL_T> &constView = constant.narrow(1, 2, 2);
        NNAssertEquals(constView.shape(), Storage<size_t>({ 10, 2 }), "const Tensor::narrow failed! Wrong shape!");
        NNAssertEquals(&constView(2, 1), &constant(2, 3), "const Tensor::narrow failed! Wrong data!");
    }

    {
        const Tensor<NN_REAL_T> &constView = constant.narrow(1, 2, 2);
        NNAssertEquals(constView.shape(), Storage<size_t>({ 10, 2 }), "const Tensor::narrow failed! Wrong shape!");
        NNAssertEquals(&constView(2, 1), &constant(2, 3), "const Tensor::narrow failed! Wrong data!");
    }

    view.resize(10, 1, 10);

    {
        const Tensor<NN_REAL_T> &constant = view;
        const Tensor<NN_REAL_T> &constView = constant.expand(1, 50);
        NNAssertEquals(constView.shape(), Storage<size_t>({ 10, 50, 10 }), "const Tensor::expand failed! Wrong shape!");
        NNAssertEquals(&constView(7, 29, 3), &constant(7, 0, 3), "const Tensor::expand failed! Wrong data!");
    }

    {
        const Tensor<NN_REAL_T> &constant = view;
        const Tensor<NN_REAL_T> &constView = constant.sub({ { 2, 4 }, { 0 }, { 7, 2 } });
        NNAssertEquals(constView.shape(), Storage<size_t>({ 4, 1, 2 }), "const Tensor::sub failed! Wrong shape!");
        NNAssertEquals(&constView(2, 0, 1), &constant(4, 0, 8), "const Tensor::sub failed! Wrong data!");
    }

    // test tensor util

    view.randn();

    empty = view + view;
    for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
        NNAssertAlmostEquals(2 * *x, *y, 1e-12, "operator+(Tensor, Tensor) failed!");

    empty += view;
    for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
        NNAssertAlmostEquals(3 * *x, *y, 1e-12, "operator+=(Tensor, Tensor) failed!");

    empty = view.copy().scale(2.5) - view;
    for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
        NNAssertAlmostEquals(1.5 * *x, *y, 1e-12, "operator-(Tensor, Tensor) failed!");

    empty -= view;
    for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
        NNAssertAlmostEquals(0.5 * *x, *y, 1e-12, "operator-=(Tensor, Tensor) failed!");

    empty = view * 2;
    for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
        NNAssertAlmostEquals(*x * 2, *y, 1e-12, "operator*(Tensor, T) failed!");

    empty *= 2;
    for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
        NNAssertAlmostEquals(*x * 4, *y, 1e-12, "operator*=(Tensor, T) failed!");

    empty = view / 2;
    for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
        NNAssertAlmostEquals(*x / 2, *y, 1e-12, "operator/(Tensor, T) failed!");

    empty /= 2;
    for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
        NNAssertAlmostEquals(*x / 4, *y, 1e-12, "operator/=(Tensor, T) failed!");

    // test serialization

    Tensor<NN_REAL_T> serializable = Tensor<NN_REAL_T>(3, 4, 5, 6).rand();
    Tensor<NN_REAL_T> serialized;

    Serialized node;
    serializable.save(node);
    serialized = Tensor<NN_REAL_T>(node);

    for(auto x = serializable.begin(), y = serialized.begin(); x != serializable.end(); ++x, ++y)
        NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::save and/or Tensor::load failed!");

    // test boundary case of forEach for high-dimensional tensors

    {
        Storage<size_t> highDims(NN_MAX_NUM_DIMENSIONS);
        for(size_t &x : highDims)
            x = 1;

        Tensor<NN_REAL_T> highDimensional(highDims, true);
        size_t highCount = 0;
        forEach([&](NN_REAL_T &x)
        {
            ++highCount;
        }, highDimensional);

        NNAssertEquals(highCount, 1, "forEach(F, Tensor<T>) failed for a high dimensional tensor!");
    }
}
