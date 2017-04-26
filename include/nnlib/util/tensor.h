#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <iomanip>
#include <memory>

#include "error.h"
#include "random.h"
#include "storage.h"
#include "algebra.h"

namespace nnlib
{

template <typename T>
class Batcher;

template <typename T>
class TensorIterator;

template <typename T>
class Tensor
{
friend class Batcher<T>;
public:
	/// Flatten a number of tensors into a vector and give the original tensors views into the new one.
	static Tensor flatten(const Storage<Tensor *> &tensors)
	{
		size_t size = 0;
		for(Tensor *t : tensors)
		{
			size += t->size();
		}
		
		Tensor flattened(size);
		size_t offset = 0;
		for(Tensor *t : tensors)
		{
			size_t i = offset;
			for(const T &value : *t)
			{
				flattened(i) = value;
				++i;
			}
			t->m_data = flattened.m_data;	// make t share data with flattened
			t->m_offset = offset;			// give t the appropriate offset in flattened
			t->resize(t->shape());			// reset strides of t to be contiguous
			offset = i;
		}
		
		return flattened;
	}
	
	/// Create a vector with the given data.
	Tensor(const Storage<T> &values) :
		m_dims({ values.size() }),
		m_strides({ 1 }),
		m_offset(0),
		m_data(new Storage<T>(values)),
		m_shared(m_data)
	{}
	
	/// Create a vector with the given data.
	Tensor(const std::initializer_list<T> &values) :
		m_dims({ values.size() }),
		m_strides({ 1 }),
		m_offset(0),
		m_data(new Storage<T>(values)),
		m_shared(m_data)
	{}
	
	/// Create a tensor with the given size and shape.
	/// \note This uses a dummy bool to differentiate this from the const Storage<T> & constructor.
	Tensor(const Storage<size_t> &dims, bool) :
		m_offset(0),
		m_data(new Storage<T>()),
		m_shared(m_data)
	{
		resize(dims);
	}
	
	/// Create a tensor with the given size and shape.
	/// \note This includes the default constructor.
	template <typename ... Ts>
	explicit Tensor(Ts... dims) :
		m_offset(0),
		m_data(new Storage<T>()),
		m_shared(m_data)
	{
		resize(dims...);
	}
	
	/// Create a tensor as a view of another tensor with the same shape.
	Tensor(Tensor &other) :
		m_dims(other.m_dims),
		m_strides(other.m_strides),
		m_offset(other.m_offset),
		m_data(other.m_data),
		m_shared(other.m_shared)
	{}
	
	/// Create a tensor as a view of another tensor with the same shape.
	Tensor(Tensor &&other) :
		m_dims(other.m_dims),
		m_strides(other.m_strides),
		m_offset(other.m_offset),
		m_data(other.m_data),
		m_shared(other.m_shared)
	{}
	
	/// Fill this with values of that.
	/// Resizes this to a 1-dimensional tensor.
	Tensor &operator=(const Storage<T> &values)
	{
		m_dims		= { values.size() };
		m_strides	= { 1 };
		m_offset	= 0;
		*m_data		= values;
		return *this;
	}
	
	/// Fill this with values of that.
	/// Resizes this to a 1-dimensional tensor.
	Tensor &operator=(const std::initializer_list<T> &values)
	{
		m_dims		= { values.size() };
		m_strides	= { 1 };
		m_offset	= 0;
		*m_data		= values;
		return *this;
	}
	
	/// Move a tensor to this.
	Tensor &operator=(Tensor &other)
	{
		m_dims		= other.m_dims;
		m_strides	= other.m_strides;
		m_offset	= other.m_offset;
		m_data		= other.m_data;
		m_shared	= other.m_shared;
		return *this;
	}
	
	/// Move a tensor to this.
	Tensor &operator=(Tensor &&other)
	{
		m_dims		= other.m_dims;
		m_strides	= other.m_strides;
		m_offset	= other.m_offset;
		m_data		= other.m_data;
		m_shared	= other.m_shared;
		return *this;
	}
	
	// MARK: Size and shape methods.
	
	/// Resize this tensor in place and, if necessary, resizes its underlying storage.
	Tensor &resize(const Storage<size_t> &dims)
	{
		// Don't allow a 0-dimensional tensor.
		if(dims.size() > 0)
		{
			m_dims = dims;
		}
		else
		{
			m_dims = { 0 };
		}
		
		m_strides.resize(m_dims.size());
		
		m_strides[m_strides.size() - 1] = 1;
		for(size_t i = m_strides.size() - 1; i > 0; --i)
		{
			m_strides[i - 1] = m_strides[i] * m_dims[i];
		}
		size_t newSize = m_offset + m_strides[0] * m_dims[0];
		if(newSize > m_data->size())
		{
			// only resize if necessary, because other tensors may share this data and need it all
			m_data->resize(newSize);
		}
		return *this;
	}
	
	/// Resize this tensor in place and, if necessary, resizes its underlying storage.
	template <typename ... Ts>
	Tensor &resize(Ts... dims)
	{
		return resize({ static_cast<size_t>(dims)... });
	}
	
	/// Creates a new tensor with a copy of this data but a new shape.
	/// The shape must be compatible.
	Tensor reshape(const Storage<size_t> &dims) const
	{
		Tensor t(dims);
		NNAssert(t.size() == size(), "Incompatible dimensions for reshaping!");
		auto k = t.begin();
		for(const T &value : *this)
		{
			*k = value;
			++k;
		}
		return t;
	}
	
	/// Creates a new tensor with a copy of this data but a new shape.
	/// The shape must be compatible.
	template <typename ... Ts>
	Tensor reshape(Ts... dims) const
	{
		return reshape({ static_cast<size_t>(dims)... });
	}
	
	/// Creates a new tensor with a subview of this data.
	/// This completely eliminates the dimth dimension.
	/// The resulting tensor has one less dimension than this.
	Tensor select(size_t dim, size_t index)
	{
		NNAssert(dim < m_dims.size(), "Narrowing dimension out of bounds!");
		NNAssert(index < m_dims[dim], "Out of dimension bounds!");
		Tensor t = *this;
		t.m_offset += index * t.m_strides[dim];
		t.m_dims.erase(dim);
		t.m_strides.erase(dim);
		return t;
	}
	
	/// Creates a new tensor with a subview of this data.
	/// This reduces the dimth dimension to size.
	/// The resulting tensor has the same number of dimensions as this.
	Tensor narrow(size_t dim, size_t index, size_t size = 1)
	{
		NNAssert(dim < m_dims.size(), "Narrowing dimension out of bounds!");
		NNAssert(index + size <= m_dims[dim], "Out of dimension bounds!");
		Tensor t = *this;
		t.m_offset = m_offset + index * m_strides[dim];
		t.m_dims[dim] = size;
		return t;
	}
	
	/// Creates a new tensor with a subview of this data.
	/// This reduces the dimth dimension to size.
	/// The resulting tensor has the same number of dimensions as this.
	const Tensor narrow(size_t dim, size_t index, size_t size = 1) const
	{
		return const_cast<Tensor *>(this)->narrow(dim, index, size);
	}
	
	/// Fills t with a subview of this data and returns t.
	/// This performs narrow on each dimension.
	/// The resulting tensor has the same number of dimensions as this.
	Tensor &sub(Tensor &t, const std::initializer_list<const std::initializer_list<size_t>> &dims)
	{
		NNAssert(dims.size() == m_dims.size(), "Invalid subtensor dimensions!");
		NNAssert(t.shape() == shape(), "Incompatible sub tensor!");
		
		size_t dim = 0;
		for(const std::initializer_list<size_t> &params : dims)
		{
			NNAssert(params.size() <= 2, "Invalid parameters for subtensor!");
			if(params.size() == 1)
			{
				size_t index = *params.begin();
				t.m_offset = t.m_offset + index * m_strides[dim];
				t.m_dims[dim] = 1;
			}
			else if(params.size() == 2)
			{
				size_t index = *params.begin();
				size_t size = *(params.begin() + 1);
				t.m_offset = t.m_offset + index * m_strides[dim];
				t.m_dims[dim] = size;
			}
			++dim;
		}
		
		return t;
	}
	
	/// Fills t with a subview of this data and returns t.
	/// This performs narrow on each dimension.
	/// The resulting tensor has the same number of dimensions as this.
	const Tensor &sub(const Tensor &t, const std::initializer_list<const std::initializer_list<size_t>> &dims) const
	{
		return const_cast<Tensor *>(this)->sub(*const_cast<Tensor *>(&t), dims);
	}
	
	/// Creates a new tensor with a subview of this data.
	/// This performs narrow on each dimension.
	/// The resulting tensor has the same number of dimensions as this.
	Tensor sub(const std::initializer_list<const std::initializer_list<size_t>> &dims)
	{
		Tensor t = *this;
		return sub(t, dims);
	}
	
	/// Creates a new tensor with a subview of this data.
	/// This performs narrow on each dimension.
	/// The resulting tensor has the same number of dimensions as this.
	const Tensor sub(const std::initializer_list<const std::initializer_list<size_t>> &dims) const
	{
		return const_cast<Tensor *>(this)->sub(dims);
	}
	
	/// Creates a new tensor with the same shape and a copy (not a view) of this data.
	Tensor copy() const
	{
		return reshape(m_dims);
	}
	
	/// Copies the shape and data from another tensor.
	Tensor &copy(const Tensor &other)
	{
		resize(other.shape());
		auto i = other.begin();
		for(T &value : *this)
		{
			value = *i;
			++i;
		}
		return *this;
	}
	
	/// Swaps data with another tensor.
	Tensor &swap(Tensor &other)
	{
		NNAssert(shape() == other.shape(), "Incompatible tensors for swapping!");
		auto i = other.begin();
		for(T &v : *this)
		{
			T t = v;
			v = *i;
			*i = t;
			++i;
		}
		return *this;
	}
	
	/// Swaps data with another tensor.
	/// Allows rvalue references, as it may be a temporary that references persistant Storage.
	Tensor &swap(Tensor &&other)
	{
		NNAssert(shape() == other.shape(), "Incompatible tensors for swapping!");
		auto i = other.begin();
		for(T &v : *this)
		{
			T t = v;
			v = *i;
			*i = t;
			++i;
		}
		return *this;
	}
	
	/// Swap two dimensions (rows and columns by default).
	/// This returns a new tensor that shares data with the original.
	Tensor transpose(size_t dim1 = 1, size_t dim2 = 0)
	{
		NNAssert(dim1 < dims() && dim2 < dims(), "Invalid dimensions for transposition!");
		Tensor t = *this;
		
		size_t temp = t.m_strides[dim1];
		t.m_strides[dim1] = t.m_strides[dim2];
		t.m_strides[dim2] = temp;
		
		temp = t.m_dims[dim1];
		t.m_dims[dim1] = t.m_dims[dim2];
		t.m_dims[dim2] = temp;
		
		return t;
	}
	
	/// Get the entire list of dimensions.
	const Storage<size_t> &shape() const
	{
		return m_dims;
	}
	
	/// Get the number of dimensions in this tensor.
	size_t dims() const
	{
		return m_dims.size();
	}
	
	/// Get the total number of elements in this tensor.
	/// \todo find a faster way, maybe cache it.
	size_t size() const
	{
		size_t result = 1;
		for(size_t s : m_dims)
		{
			result *= s;
		}
		return result;
	}
	
	/// Get the size of a given dimension.
	size_t size(size_t dim) const
	{
		NNAssert(dim < m_dims.size(), "Invalid dimension!");
		return m_dims[dim];
	}
	
	/// Get the stride of a given dimension.
	size_t stride(size_t dim) const
	{
		return m_strides[dim];
	}
	
	// MARK: Element manipulation methods.
	
	/// Set every element in this tensor to value.
	Tensor &fill(const T &value)
	{
		std::fill(begin(), end(), value);
		return *this;
	}
	
	/// Set every element in this tensor to 0.
	Tensor &zeros()
	{
		return fill(0);
	}
	
	/// Set every element in this tensor to 1.
	Tensor &ones()
	{
		return fill(1);
	}
	
	/// Set every element in this tensor to a uniformly distributed random value.
	Tensor &rand(const T &from = -1, const T &to = 1)
	{
		for(T &v : *this)
		{
			v = Random<T>::uniform(from, to);
		}
		return *this;
	}
	
	/// Set every element in this tensor to a normally distributed random value.
	Tensor &randn(const T &mean = 0, const T &stddev = 1)
	{
		for(T &v : *this)
		{
			v = Random<T>::normal(mean, stddev);
		}
		return *this;
	}
	
	/// Set every element in this tensor to a normally distributed random value, capped.
	Tensor &randn(const T &mean, const T &stddev, const T &cap)
	{
		for(T &v : *this)
		{
			v = Random<T>::normal(mean, stddev, cap);
		}
		return *this;
	}
	
	/// Multiply this tensor by a scalar.
	Tensor &scale(T alpha)
	{
		for(T &v : *this)
		{
			v *= alpha;
		}
		return *this;
	}
	
	// MARK: Algebra
	
	Tensor &multiplyMM(const Tensor<T> &A, const Tensor<T> &B, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		Algebra<T>::multiplyMM(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			B.ptr(), B.size(0), B.size(1), B.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMTM(const Tensor<T> &A, const Tensor<T> &B, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		Algebra<T>::multiplyMTM(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			B.ptr(), B.size(0), B.size(1), B.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMMT(const Tensor<T> &A, const Tensor<T> &B, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		Algebra<T>::multiplyMMT(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			B.ptr(), B.size(0), B.size(1), B.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMTMT(const Tensor<T> &A, const Tensor<T> &B, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		Algebra<T>::multiplyMTMT(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			B.ptr(), B.size(0), B.size(1), B.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMV(const Tensor<T> &A, const Tensor<T> &x, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && x.dims() == 1 && dims() == 1, "Incompatible operands!");
		Algebra<T>::multiplyMV(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			x.ptr(), x.size(), x.stride(0),
			ptr(), size(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMTV(const Tensor<T> &A, const Tensor<T> &x, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && x.dims() == 1 && dims() == 1, "Incompatible operands!");
		Algebra<T>::multiplyMTV(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			x.ptr(), x.size(), x.stride(0),
			ptr(), size(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyVTV(const Tensor<T> &x, const Tensor<T> &y, T alpha = 1)
	{
		NNAssert(x.dims() == 1 && y.dims() == 1 && dims() == 2, "Incompatible operands!");
		Algebra<T>::multiplyVTV(
			x.ptr(), x.size(), x.stride(0),
			y.ptr(), y.size(), y.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha
		);
		return *this;
	}
	
	Tensor &addVV(const Tensor<T> &x, T alpha = 1)
	{
		NNAssert(x.dims() == 1 && dims() == 1 && x.size() == size(), "Incompatible operands!");
		Algebra<T>::addVV(
			x.ptr(), x.size(), x.stride(0),
			ptr(), size(), stride(0),
			alpha
		);
		return *this;
	}
	
	Tensor &addMM(const Tensor<T> &A, T alpha = 1)
	{
		NNAssert(A.dims() == 2 && dims() == 2 && A.shape() == shape(), "Incompatible operands!");
		for(size_t i = 0, end = size(0); i < end; ++i)
		{
			Algebra<T>::addVV(
				A.ptr() + i * A.stride(0), A.size(1), A.stride(1),
				ptr() + i * stride(0), size(1), stride(1),
				alpha
			);
		}
		return *this;
	}
	
	// MARK: Statistical methods
	
	T sum() const
	{
		T result = 0;
		for(const T &v : *this)
		{
			result += v;
		}
		return result;
	}
	
	T min() const
	{
		T result = *begin();
		for(const T &v : *this)
		{
			if(v < result)
			{
				result = v;
			}
		}
		return result;
	}
	
	T max() const
	{
		T result = *begin();
		for(const T &v : *this)
		{
			if(v > result)
			{
				result = v;
			}
		}
		return result;
	}
	
	// MARK: Element/data access methods.
	
	/// Element access given a multidimensional index.
	T &operator()(const Storage<size_t> &indices)
	{
		return (*m_data)[indexOf(indices)];
	}
	
	/// Element access given a multidimensional index.
	const T &operator()(const Storage<size_t> &indices) const
	{
		return (*m_data)[indexOf(indices)];
	}
	
	/// Element access given a multidimensional index.
	template <typename ... Ts>
	T &operator()(Ts... indices)
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}
	
	/// Element access given a multidimensional index.
	template <typename ... Ts>
	const T &operator()(Ts... indices) const
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}
	
	/// Direct raw pointer access.
	T *ptr()
	{
		return m_data->ptr() + m_offset;
	}
	
	/// Direct raw pointer access.
	const T *ptr() const
	{
		return m_data->ptr() + m_offset;
	}
	
	/// Direct storage access.
	Storage<T> &storage()
	{
		return *m_data;
	}
	
	/// Direct storage access.
	const Storage<T> &storage() const
	{
		return *m_data;
	}
	
	// MARK: Iterators
	
	TensorIterator<T> begin()
	{
		return TensorIterator<T>(this);
	}
	
	TensorIterator<T> end()
	{
		return TensorIterator<T>(this, true);
	}
	
	TensorIterator<const T> begin() const
	{
		return TensorIterator<const T>(this);
	}
	
	TensorIterator<const T> end() const
	{
		return TensorIterator<const T>(this, true);
	}
	
private:
	Storage<size_t> m_dims;					///< The length along each dimension.
	Storage<size_t> m_strides;				///< Strides between dimensions.
	size_t m_offset;						///< Offset of data for this view.
	Storage<T> *m_data;						///< The actual data.
	std::shared_ptr<Storage<T>> m_shared;	///< Wrapped around m_data for ARC.
	
	/// Get the appropriate contiguous index given the multidimensional index.
	size_t indexOf(const Storage<size_t> &indices) const
	{
		NNAssert(indices.size() == m_dims.size(), "Incorrect number of dimensions!");
		size_t sum = m_offset;
		for(size_t i = 0, j = indices.size(); i < j; ++i)
		{
			sum += indices[i] * m_strides[i];
		}
		NNAssert(sum < m_data->size(), "Index out of bounds!");
		return sum;
	}
};

template <typename T>
class TensorIterator : public std::iterator<std::forward_iterator_tag, T, T, const T *, T &>
{
using TT = typename std::remove_const<T>::type;
public:
	TensorIterator(const Tensor<TT> *tensor, bool end = false) :
		m_tensor(const_cast<Tensor<TT> *>(tensor)),
		m_indices(tensor->dims(), 0)
	{
		if(end)
		{
			m_indices[0] = m_tensor->size(0);
		}
	}
	
	TensorIterator &operator++()
	{
		size_t dim = m_indices.size() - 1;
		++m_indices[dim];
		while(m_indices[dim] >= m_tensor->size(dim) && dim > 0)
		{
			m_indices[dim] = 0;
			--dim;
			++m_indices[dim];
		}
		return *this;
	}
	
	T &operator*()
	{
		return (*m_tensor)(m_indices);
	}
	
	bool operator==(const TensorIterator &other)
	{
		if(m_tensor != other.m_tensor)
		{
			return false;
		}
		
		for(size_t i = 0, j = m_indices.size(); i < j; ++i)
		{
			if(m_indices[i] != other.m_indices[i])
			{
				return false;
			}
		}
		
		return true;
	}
	
	bool operator !=(const TensorIterator &other)
	{
		return !(*this == other);
	}
private:
	Tensor<TT> *m_tensor;
	Storage<size_t> m_indices;
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const Tensor<T> &t)
{
	out << std::left << std::setprecision(5) << std::fixed;
	
	if(t.dims() == 1)
	{
		for(size_t i = 0; i < t.size(0); ++i)
		{
			out << t(i) << "\n";
		}
	}
	else if(t.dims() == 2)
	{
		for(size_t i = 0; i < t.size(0); ++i)
		{
			for(size_t j = 0; j < t.size(1); ++j)
			{
				out << std::setw(10) << t(i, j);
			}
			out << "\n";
		}
	}
	
	out << "[ Tensor of dimension " << t.size(0);
	for(size_t i = 1; i < t.dims(); ++i)
	{
		out << " x " << t.size(i);
	}
	out << " ]";
	
	return out;
}

}

#endif
