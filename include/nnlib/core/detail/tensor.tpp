#ifndef CORE_TENSOR_TPP
#define CORE_TENSOR_TPP

#include "../tensor.hpp"
#include "nnlib/util/random.hpp"

namespace nnlib
{

template <typename T>
Tensor<T> Tensor<T>::vectorize(const Storage<Tensor<T> *> &tensors)
{
    size_t size = 0;
    for(Tensor<T> *t : tensors)
        size += t->size();

    if(isVectorized(tensors))
    {
        Tensor<T> flattened = *tensors[0];
        flattened.m_dims = { size };
        flattened.m_strides = { 1 };
        flattened.m_size = size;
        return flattened;
    }

    Tensor<T> flattened(size);

    size_t offset = 0;
    for(Tensor<T> *t : tensors)
    {
        Tensor<T> view = flattened.narrow(0, offset, t->size());
        view.copy(*t);
        offset += t->size();
        *t = view.view(t->shape());
    }

    return flattened;
}

template <typename T>
Tensor<T> Tensor<T>::concatenate(const Storage<Tensor<T> *> &tensors, size_t dim)
{
    if(tensors.size() == 0)
        return Tensor();

    dim = std::min(dim, tensors[0]->dims() - 1);

    // Check compatibility and calculate result dimensions

    Storage<size_t> dims = tensors[0]->shape();
    for(size_t i = 1, count = tensors.size(); i < count; ++i)
    {
        NNAssertEquals(tensors[i]->select(dim, 0).shape(), tensors[0]->select(dim, 0).shape(), "Incompatible tensors for concatenation along the given dimension!");
        dims[dim] += tensors[i]->size(dim);
    }

    // Create the shared tensor

    Tensor<T> concatenated(dims, true);

    // Copy data from tensors into the shared tensor, then give tensors views into the shared tensor

    size_t dimOffset = 0;
    for(Tensor<T> *t : tensors)
    {
        Tensor<T> view = concatenated.narrow(dim, dimOffset, t->size(dim));
        view.copy(*t);
        dimOffset += t->size(dim);
        *t = view;
    }

    return concatenated;
}

template <typename T>
Tensor<T>::Tensor() :
    m_dims({ 0 }),
    m_strides({ 1 }),
    m_offset(0),
    m_data(new Storage<T>()),
    m_shared(m_data),
    m_size(0),
    m_contiguous(true)
{}

template <typename T>
Tensor<T>::Tensor(const Storage<T> &values) :
    m_dims({ values.size() }),
    m_strides({ 1 }),
    m_offset(0),
    m_data(new Storage<T>(values)),
    m_shared(m_data),
    m_size(values.size()),
    m_contiguous(true)
{}

template <typename T>
Tensor<T>::Tensor(const std::initializer_list<T> &values) :
    m_dims({ values.size() }),
    m_strides({ 1 }),
    m_offset(0),
    m_data(new Storage<T>(values)),
    m_shared(m_data),
    m_size(values.size()),
    m_contiguous(true)
{}

template <typename T>
Tensor<T>::Tensor(const Storage<size_t> &dims, bool) :
    m_offset(0),
    m_data(new Storage<T>()),
    m_shared(m_data)
{
    resize(dims);
}

template <typename T>
Tensor<T>::Tensor(Tensor<T> &other) :
    m_dims(other.m_dims),
    m_strides(other.m_strides),
    m_offset(other.m_offset),
    m_data(other.m_data),
    m_shared(other.m_shared),
    m_size(other.m_size),
    m_contiguous(other.m_contiguous)
{}

template <typename T>
Tensor<T>::Tensor(Tensor<T> &&other) :
    m_dims(other.m_dims),
    m_strides(other.m_strides),
    m_offset(other.m_offset),
    m_data(other.m_data),
    m_shared(other.m_shared),
    m_size(other.m_size),
    m_contiguous(other.m_contiguous)
{}

template <typename T>
Tensor<T>::Tensor(const Serialized &node) :
    m_offset(0),
    m_data(new Storage<T>()),
    m_shared(m_data)
{
    if(node.type() == Serialized::Object)
    {
        resize(node.get<Storage<size_t>>("dims"));
        node.get("data", begin(), end());
    }
    else
    {
        resize(node.size(), node.get(0)->size());
        for(size_t i = 0; i < node.size(); ++i)
            for(size_t j = 0; j < node.get(0)->size(); ++j)
                at(i, j) = node.get(i)->get<T>(j);
    }
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Storage<T> &values)
{
    m_dims			= { values.size() };
    m_strides		= { 1 };
    m_offset		= 0;
    *m_data			= values;
    m_size			= values.size();
    m_contiguous	= true;
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const std::initializer_list<T> &values)
{
    m_dims			= { values.size() };
    m_strides		= { 1 };
    m_offset		= 0;
    *m_data			= values;
    m_size			= values.size();
    m_contiguous	= true;
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &other)
{
    m_dims			= other.m_dims;
    m_strides		= other.m_strides;
    m_offset		= other.m_offset;
    m_data			= other.m_data;
    m_shared		= other.m_shared;
    m_size			= other.m_size;
    m_contiguous	= other.m_contiguous;
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &&other)
{
    m_dims			= other.m_dims;
    m_strides		= other.m_strides;
    m_offset		= other.m_offset;
    m_data			= other.m_data;
    m_shared		= other.m_shared;
    m_size			= other.m_size;
    m_contiguous	= other.m_contiguous;
    return *this;
}

template <typename T>
bool Tensor<T>::shared() const
{
    return m_shared.use_count() > 1;
}

template <typename T>
bool Tensor<T>::sharedWith(const Tensor<T> &other) const
{
    return m_data == other.m_data;
}

template <typename T>
bool Tensor<T>::sharedWith(const Storage<Tensor<T> *> &tensors) const
{
    for(size_t i = 0, count = tensors.size(); i < count; ++i)
        if(!sharedWith(*tensors[i]))
            return false;
    return true;
}

template <typename T>
size_t Tensor<T>::sharedCount() const
{
    return m_shared.use_count();
}

template <typename T>
Tensor<T> &Tensor<T>::resize(const Storage<size_t> &dims)
{
    NNHardAssert(dims.size() > 0, "Cannot create a zero-dimensional tensor!");

    if(dims == m_dims)
        return *this;

    // Calculate new strides and size.

    Storage<size_t> strides(dims.size());
    strides.back() = 1;
    for(size_t i = strides.size() - 1; i > 0; --i)
        strides[i - 1] = strides[i] * dims[i];

    size_t size = strides[0] * dims[0];

    // Resize underlying storage. If not unique and this is smaller or not contiguous, break shared connection.

    if(shared() && (size > m_size || !m_contiguous))
        *this = Tensor(*m_data).resize(dims);
    else
    {
        if(!shared())
            m_data->resize(m_offset + size);
        m_dims = std::move(dims);
        m_strides = std::move(strides);
        m_size = size;
        m_contiguous = true;
    }

    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::resizeDim(size_t dim, size_t size)
{
    if(m_dims[dim] == size)
        return *this;

    Storage<size_t> dims = m_dims;
    dims[dim] = size;

    return resize(dims);
}

template <typename T>
Tensor<T> Tensor<T>::view(const Storage<size_t> &dims)
{
    NNHardAssert(m_contiguous, "Expected a contiguous tensor!");

    size_t size = 1;
    for(size_t d : dims)
        size *= d;
    NNHardAssertLessThanOrEquals(size, m_size, "Expected view to be smaller than the original tensor!");

    Tensor<T> t = *this;
    return t.resize(dims);
}

template <typename T>
const Tensor<T> Tensor<T>::view(const Storage<size_t> &dims) const
{
    NNHardAssert(m_contiguous, "Expected a contiguous tensor!");

    size_t size = 1;
    for(size_t d : dims)
        size *= d;
    NNHardAssertLessThanOrEquals(size, m_size, "Expected view to be smaller than the original tensor!");

    Tensor<T> t = *const_cast<Tensor<T> *>(this);
    return t.resize(dims);
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const Storage<size_t> &dims) const
{
    Tensor<T> t(dims, true);
    NNAssertEquals(t.size(), size(), "Incompatible dimensions for reshaping!");
    auto k = t.begin();
    forEach([&](const T &value)
    {
        *k = value;
        ++k;
    }, *this);
    return t;
}

template <typename T>
Tensor<T> Tensor<T>::select(size_t dim, size_t index)
{
    NNAssertLessThan(dim, m_dims.size(), "Narrowing dimension out of bounds!");
    NNAssertLessThan(index, m_dims[dim], "Out of dimension bounds!");
    Tensor<T> t = *this;
    t.m_offset += index * t.m_strides[dim];
    t.m_dims.erase(dim);
    t.m_strides.erase(dim);
    t.recalculateSize();
    t.checkContiguous();
    return t;
}

template <typename T>
const Tensor<T> Tensor<T>::select(size_t dim, size_t index) const
{
    return const_cast<Tensor<T> *>(this)->select(dim, index);
}

template <typename T>
Tensor<T> Tensor<T>::narrow(size_t dim, size_t index, size_t size)
{
    NNAssertLessThan(dim, m_dims.size(), "Narrowing dimension out of bounds!");
    NNAssertLessThanOrEquals(index + size, m_dims[dim], "Out of dimension bounds!");
    Tensor<T> t = *this;
    t.m_offset = m_offset + index * m_strides[dim];
    t.m_dims[dim] = size;
    t.recalculateSize();
    t.checkContiguous();
    return t;
}

template <typename T>
const Tensor<T> Tensor<T>::narrow(size_t dim, size_t index, size_t size) const
{
    return const_cast<Tensor<T> *>(this)->narrow(dim, index, size);
}

template <typename T>
Tensor<T> Tensor<T>::expand(size_t dim, size_t size)
{
    NNAssertLessThan(dim, m_dims.size(), "Expanding dimension out of bounds!");
    NNAssertEquals(m_dims[dim], 1, "Can only expand a dimension of size 1!");
    Tensor<T> t = *this;
    t.m_dims[dim] = size;
    t.m_strides[dim] = 0;
    t.recalculateSize();
    t.checkContiguous();
    return t;
}

template <typename T>
const Tensor<T> Tensor<T>::expand(size_t dim, size_t size) const
{
    return const_cast<Tensor<T> *>(this)->expand(dim, size);
}

template <typename T>
Tensor<T> &Tensor<T>::sub(Tensor<T> &t, const std::initializer_list<const std::initializer_list<size_t>> &dims)
{
    NNAssertEquals(dims.size(), m_dims.size(), "Invalid subtensor dimensions!");
    t = *this;

    size_t dim = 0;
    for(const std::initializer_list<size_t> &params : dims)
    {
        NNAssertLessThanOrEquals(params.size(), 2, "Invalid parameters for subtensor!");
        if(params.size() == 1)
        {
            size_t index = *params.begin();
            NNAssertLessThan(index, m_dims[dim], "Incompatible index!");
            t.m_offset += index * m_strides[dim];
            t.m_dims[dim] = 1;
        }
        else if(params.size() == 2)
        {
            size_t index = *params.begin();
            size_t size = *(params.begin() + 1);
            NNAssertLessThanOrEquals(index + size, m_dims[dim], "Incompatible index and size!");
            t.m_offset += index * m_strides[dim];
            t.m_dims[dim] = size;
        }
        ++dim;
    }

    t.recalculateSize();
    t.checkContiguous();

    return t;
}

template <typename T>
Tensor<T> Tensor<T>::sub(const std::initializer_list<const std::initializer_list<size_t>> &dims)
{
    Tensor<T> t = *this;
    return sub(t, dims);
}

template <typename T>
const Tensor<T> Tensor<T>::sub(const std::initializer_list<const std::initializer_list<size_t>> &dims) const
{
    return const_cast<Tensor<T> *>(this)->sub(dims);
}

template <typename T>
Tensor<T> Tensor<T>::copy() const
{
    return reshape(m_dims);
}

template <typename T>
Tensor<T> &Tensor<T>::copy(const Tensor<T> &other)
{
    NNAssertEquals(size(), other.size(), "Incompatible tensor for copying!");
    auto i = other.begin();
    forEach([&](T &value)
    {
        value = *i;
        ++i;
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::swap(Tensor<T> &other)
{
    NNAssertEquals(shape(), other.shape(), "Incompatible tensors for swapping!");
    auto i = other.begin();
    forEach([&](T &v)
    {
        T t = v;
        v = *i;
        *i = t;
        ++i;
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::swap(Tensor<T> &&other)
{
    NNAssertEquals(shape(), other.shape(), "Incompatible tensors for swapping!");
    auto i = other.begin();
    forEach([&](T &v)
    {
        T t = v;
        v = *i;
        *i = t;
        ++i;
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::transpose(size_t dim1, size_t dim2)
{
    NNAssertLessThan(dim1, dims(), "Invalid dimensions for transposition!");
    NNAssertLessThan(dim2, dims(), "Invalid dimensions for transposition!");
    Tensor<T> t = *this;

    size_t temp = t.m_strides[dim1];
    t.m_strides[dim1] = t.m_strides[dim2];
    t.m_strides[dim2] = temp;

    temp = t.m_dims[dim1];
    t.m_dims[dim1] = t.m_dims[dim2];
    t.m_dims[dim2] = temp;

    t.checkContiguous();

    return t;
}

template <typename T>
const Tensor<T> Tensor<T>::transpose(size_t dim1, size_t dim2) const
{
    return const_cast<Tensor<T> &>(*this).transpose(dim1, dim2);
}

template <typename T>
const Storage<size_t> &Tensor<T>::shape() const
{
    return m_dims;
}

template <typename T>
const Storage<size_t> &Tensor<T>::strides() const
{
    return m_strides;
}

template <typename T>
size_t Tensor<T>::dims() const
{
    return m_dims.size();
}

template <typename T>
size_t Tensor<T>::size() const
{
    return m_size;
}

template <typename T>
size_t Tensor<T>::size(size_t dim) const
{
    NNAssertLessThan(dim, m_dims.size(), "Invalid dimension!");
    return m_dims[dim];
}

template <typename T>
bool Tensor<T>::contiguous() const
{
    return m_contiguous;
}

template <typename T>
Tensor<T> &Tensor<T>::makeContiguous()
{
    if(!m_contiguous)
        *this = copy();
    return *this;
}

template <typename T>
size_t Tensor<T>::stride(size_t dim) const
{
    return m_strides[dim];
}

template <typename T>
Tensor<T> &Tensor<T>::fill(const T &value)
{
    forEach([&](T &v)
    {
        v = value;
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::zeros()
{
    return fill(0);
}

template <typename T>
Tensor<T> &Tensor<T>::ones()
{
    return fill(1);
}

template <typename T>
Tensor<T> &Tensor<T>::rand(const T &from, const T &to)
{
    forEach([&](T &v)
    {
        v = Random<T>::sharedRandom().uniform(from, to);
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::randn(const T &mean, const T &stddev)
{
    forEach([&](T &v)
    {
        v = Random<T>::sharedRandom().normal(mean, stddev);
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::bernoulli(const T &p)
{
    forEach([&](T &v)
    {
        v = Random<T>::sharedRandom().bernoulli(p);
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::randn(const T &mean, const T &stddev, const T &cap)
{
    forEach([&](T &v)
    {
        v = Random<T>::sharedRandom().normal(mean, stddev, cap);
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::scale(T alpha)
{
    forEach([&](T &v)
    {
        v *= alpha;
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::add(T alpha)
{
    forEach([&](T &v)
    {
        v += alpha;
    }, *this);
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::sparsify(T epsilon) const
{
    size_t count = 0;
    forEach([&](T x)
    {
        if(std::abs(x) > epsilon)
            ++count;
    }, *this);

    Tensor<T> sparse(count + 1, m_dims.size() + 1);
    for(size_t i = 0; i < m_dims.size(); ++i)
        sparse(0, i) = m_dims[i];

    Storage<size_t> indices(m_dims.size());
    size_t idx = 1;

    while(indices.front() < m_dims[0])
    {
        const T &x = (*this)(indices);

        if(std::abs(x) > epsilon)
        {
            for(size_t j = 0, jend = indices.size(); j != jend; ++j)
                sparse(idx, j) = indices[j];
            sparse(idx, indices.size()) = x;
            ++idx;
        }

        size_t d = indices.size() - 1;
        ++indices[d];

        while(indices[d] >= m_dims[d] && d > 0)
        {
            indices[d] = 0;
            --d;
            ++indices[d];
        }
    }

    return sparse;
}

template <typename T>
Tensor<T> Tensor<T>::unsparsify() const
{
    NNAssertEquals(m_dims.size(), 2, "Sparse tensors must be represented by matrices!");

    Storage<size_t> dims(m_dims[1] - 1);
    for(size_t i = 0, end = dims.size(); i != end; ++i)
        dims[i] = (*this)(0, i);

    Tensor<T> dense(dims, true);
    dense.fill(0);

    for(size_t i = 1, end = m_dims[0], jend = m_dims[1] - 1; i != end; ++i)
    {
        for(size_t j = 0; j != jend; ++j)
            dims[j] = (*this)(i, j);
        dense(dims) = (*this)(i, jend);
    }

    return dense;
}

template <typename T>
T &Tensor<T>::at(const Storage<size_t> &indices)
{
    return (*m_data)[indexOf(indices)];
}

template <typename T>
const T &Tensor<T>::at(const Storage<size_t> &indices) const
{
    return (*m_data)[indexOf(indices)];
}

template <typename T>
T &Tensor<T>::operator()(const Storage<size_t> &indices)
{
    return (*m_data)[indexOf(indices)];
}

template <typename T>
const T &Tensor<T>::operator()(const Storage<size_t> &indices) const
{
    return (*m_data)[indexOf(indices)];
}

template <typename T>
T *Tensor<T>::ptr()
{
    return m_data->ptr() + m_offset;
}

template <typename T>
const T *Tensor<T>::ptr() const
{
    return m_data->ptr() + m_offset;
}

template <typename T>
Storage<T> &Tensor<T>::data()
{
    return *m_data;
}

template <typename T>
const Storage<T> &Tensor<T>::data() const
{
    return *m_data;
}

template <typename T>
TensorIterator<T> Tensor<T>::begin()
{
    return TensorIterator<T>(this);
}

template <typename T>
TensorIterator<T> Tensor<T>::end()
{
    return TensorIterator<T>(this, true);
}

template <typename T>
TensorIterator<const T> Tensor<T>::begin() const
{
    return TensorIterator<const T>(this);
}

template <typename T>
TensorIterator<const T> Tensor<T>::end() const
{
    return TensorIterator<const T>(this, true);
}

template <typename T>
void Tensor<T>::save(Serialized &node) const
{
    node.set("dims", m_dims);
    node.set("data", begin(), end());
}

template <typename T>
bool Tensor<T>::isVectorized(const Storage<Tensor<T> *> &tensors)
{
    Tensor<T> *prev = nullptr;
    for(Tensor<T> *t : tensors)
    {
        if(!t->m_contiguous || (prev != nullptr && (!t->sharedWith(*prev) || prev->ptr() + prev->size() != t->ptr())))
            return false;
        prev = t;
    }
    return true;
}

template <typename T>
size_t Tensor<T>::indexOf(const std::initializer_list<size_t> &indices) const
{
    NNAssertEquals(indices.size(), m_dims.size(), "Incorrect number of dimensions!");
    size_t i = 0, sum = m_offset;
    for(size_t idx : indices)
        sum += idx * m_strides[i++];

    NNAssertLessThan(sum, m_data->size(), "Index out of bounds!");
    return sum;
}

template <typename T>
size_t Tensor<T>::indexOf(const Storage<size_t> &indices) const
{
    NNAssertEquals(indices.size(), m_dims.size(), "Incorrect number of dimensions!");
    size_t sum = m_offset;
    for(size_t i = 0, j = indices.size(); i < j; ++i)
        sum += indices[i] * m_strides[i];

    NNAssertLessThan(sum, m_data->size(), "Index out of bounds!");
    return sum;
}

template <typename T>
void Tensor<T>::recalculateSize()
{
    m_size = 1;
    for(size_t s : m_dims)
        m_size *= s;
}

template <typename T>
void Tensor<T>::checkContiguous()
{
    m_contiguous = true;

    size_t stride = 1;
    for(size_t i = m_dims.size(); m_contiguous && i > 0; --i)
    {
        if(m_strides[i - 1] != stride)
            m_contiguous = false;
        stride *= m_dims[i - 1];
    }
}

}

#endif
