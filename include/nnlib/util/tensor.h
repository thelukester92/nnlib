#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <iomanip>
#include <memory>

#include "error.h"
#include "random.h"
#include "storage.h"
#include "algebra.h"
#include "archive.h"

namespace nnlib
{

class Tensor;

/// Iterator for the tensor class.
/// This is templated so that it can return const real_t or real_t.
template <typename T>
class TensorIterator : public std::iterator<std::forward_iterator_tag, T, T, const T *, T &>
{
public:
	TensorIterator(const Tensor *tensor, bool end = false);
	TensorIterator &advance(size_t dim = (size_t) -1);
	TensorIterator &operator++();
	TensorIterator operator++(int);
	T &operator*();
	bool operator==(const TensorIterator &other);
	bool operator!=(const TensorIterator &other);
private:
	Tensor *m_tensor;
	Storage<size_t> m_indices;
};

/// \brief The standard input and output type in nnlib.
///
/// A tensor can be a vector (one dimension), a matrix (two dimensions), or a higher-order tensor.
/// Tensors provide views into Storage objects, and multiple tensors can share the same Storage.
class Tensor
{
public:
	/// \brief Flattens a number of tensors into a vector.
	///
	/// Each tensor in the parameter becomes a subview into a single shared Storage.
	/// \param tensors A list of tensors to flatten.
	/// \return The flattened tensor.
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
			for(const real_t &value : *t)
			{
				flattened(i) = value;
				++i;
			}
			t->m_data = flattened.m_data;		// make t share data with flattened
			t->m_shared = flattened.m_shared;	// ARC
			t->m_offset = offset;				// give t the appropriate offset in flattened
			t->resize(t->shape());				// reset strides of t to be contiguous
			offset = i;
		}
		
		return flattened;
	}
	
	/// \brief Create a tensor with the given data.
	///
	/// This performs a deep copy of the data given in the parameter.
	/// The constructed tensor is one-dimensional (a vector).
	/// \param values A Storage containing the values to store in the new tensor.
	Tensor(const Storage<real_t> &values) :
		m_dims({ values.size() }),
		m_strides({ 1 }),
		m_offset(0),
		m_data(new Storage<real_t>(values)),
		m_shared(m_data)
	{}
	
	/// \brief Create a tensor with the given data.
	///
	/// This performs a deep copy of the data given in the parameter.
	/// The constructed tensor is one-dimensional (a vector).
	/// \param values An initializer_list containing the values to store in the new tensor.
	Tensor(const std::initializer_list<real_t> &values) :
		m_dims({ values.size() }),
		m_strides({ 1 }),
		m_offset(0),
		m_data(new Storage<real_t>(values)),
		m_shared(m_data)
	{}
	
	/// \brief Create a tensor with the given shape.
	///
	/// This creates an n-dimensional tensor where n is the size of the input parameter.
	/// \param dims A Storage containing the dimension sizes for the new tensor.
	/// \note This contructor uses a dummy bool to differentiate itself from the const Storage<real_t> & constructor. This is important when T = size_t.
	Tensor(const Storage<size_t> &dims, bool) :
		m_offset(0),
		m_data(new Storage<real_t>()),
		m_shared(m_data)
	{
		resize(dims);
	}
	
	/// \brief Create a tensor with the given shape.
	///
	/// This creates an n-dimensional tensor where n is the size of the input parameter.
	/// \param dims A parameter pack containing the dimension sizes for the new tensor.
	/// \note This is the default constructor when `sizeof...(dims) == 0`.
	template <typename ... Ts>
	explicit Tensor(Ts... dims) :
		m_offset(0),
		m_data(new Storage<real_t>()),
		m_shared(m_data)
	{
		resize(dims...);
	}
	
	/// \brief Create a tensor as a view of another tensor with the same shape.
	///
	/// The new tensor shares the parameter's storage and copies the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	/// \note This is not a copy constructor. It essentially performs a shallow copy.
	Tensor(Tensor &other) :
		m_dims(other.m_dims),
		m_strides(other.m_strides),
		m_offset(other.m_offset),
		m_data(other.m_data),
		m_shared(other.m_shared)
	{}
	
	/// \brief Move constructor for a tensor.
	///
	/// The new tensor shares the parameter's storage and copies the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	Tensor(Tensor &&other) :
		m_dims(other.m_dims),
		m_strides(other.m_strides),
		m_offset(other.m_offset),
		m_data(other.m_data),
		m_shared(other.m_shared)
	{}
	
	/// \brief Replace tensor contents with new values.
	///
	/// Resizes the tensor to be a vector (one-dimensional) and copies data from values.
	/// \param values A Storage containing the values to store in the new tensor.
	Tensor &operator=(const Storage<real_t> &values)
	{
		m_dims		= { values.size() };
		m_strides	= { 1 };
		m_offset	= 0;
		*m_data		= values;
		return *this;
	}
	
	/// \brief Replace tensor contents with new values.
	///
	/// Resizes the tensor to be a vector (one-dimensional) and copies data from values.
	/// \param values An initializer_list containing the values to store in the new tensor.
	Tensor &operator=(const std::initializer_list<real_t> &values)
	{
		m_dims		= { values.size() };
		m_strides	= { 1 };
		m_offset	= 0;
		*m_data		= values;
		return *this;
	}
	
	/// \brief Make this tensor a view of another tensor and copy its shape.
	///
	/// This tensor will share the parameter's storage and copy the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	/// \note This essentially performs a shallow copy.
	Tensor &operator=(Tensor &other)
	{
		m_dims		= other.m_dims;
		m_strides	= other.m_strides;
		m_offset	= other.m_offset;
		m_data		= other.m_data;
		m_shared	= other.m_shared;
		return *this;
	}
	
	/// \brief Move assignment for a tensor.
	///
	/// This tensor will share the parameter's storage and copy the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	/// \note This essentially performs a shallow copy.
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
	
	/// \brief Resize this tensor in place and, if necessary, resize its underlying storage.
	///
	/// \param dims The new shape for the tensor.
	/// \return The tensor, for chaining.
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
	
	/// \brief Resize this tensor in place and, if necessary, resize its underlying storage.
	///
	/// \param dims A parameter pack containing the new shape for the tensor. Must not be empty.
	/// \return The tensor, for chaining.
	template <typename ... Ts>
	Tensor &resize(Ts... dims)
	{
		return resize({ static_cast<size_t>(dims)... });
	}
	
	/// \brief Resize one dimension of this tensor in place and, if necessary, resize its underlying storage.
	///
	/// \param dim Which dimension to resize.
	/// \param size The new size of the given dimension.
	/// \return The tensor, for chaining.
	Tensor &resizeDim(size_t dim, size_t size)
	{
		m_dims[dim] = size;
		return resize(m_dims);
	}
	
	/// \brief Creates a new tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// \param dims A Storage containing the new shape.
	/// \return A tensor that views the same storage as this tensor.
	Tensor view(const Storage<size_t> &dims)
	{
		Tensor t = *this;
		return t.resize(dims);
	}
	
	/// \brief Creates a new tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// \param dims A parameter pack containing the new shape.
	/// \return A tensor that views the same storage as this tensor.
	template <typename ... Ts>
	Tensor view(Ts... dims)
	{
		return view({ static_cast<size_t>(dims)... });
	}
	
	/// \brief Creates a new constant tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// \param dims A Storage containing the new shape.
	/// \return A constant tensor that views the same storage as this tensor.
	const Tensor view(const Storage<size_t> &dims) const
	{
		Tensor t = *const_cast<Tensor *>(this);
		return t.resize(dims);
	}
	
	/// \brief Creates a new constant tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// \param dims A parameter pack containing the new shape.
	/// \return A constant tensor that views the same storage as this tensor.
	template <typename ... Ts>
	const Tensor view(Ts... dims) const
	{
		return view({ static_cast<size_t>(dims)... });
	}
	
	/// \brief Creates a new tensor with a copy of this tensor's data and a new shape.
	///
	/// This performs a deep copy of the data.
	/// The given shape must be compatible; that is, the resulting tensor must have as much data as this tensor.
	/// \param dims A Storage containing the new shape.
	/// \return A tensor that with the given shape and a copy of the data in this tensor.
	Tensor reshape(const Storage<size_t> &dims) const
	{
		Tensor t(dims);
		NNAssert(t.size() == size(), "Incompatible dimensions for reshaping!");
		auto k = t.begin();
		for(const real_t &value : *this)
		{
			*k = value;
			++k;
		}
		return t;
	}
	
	/// \brief Creates a new tensor with a copy of this tensor's data and a new shape.
	///
	/// This performs a deep copy of the data.
	/// The given shape must be compatible; that is, the resulting tensor must have as much data as this tensor.
	/// \param dims A parameter pack containing the new shape.
	/// \return A tensor that with the given shape and a copy of the data in this tensor.
	template <typename ... Ts>
	Tensor reshape(Ts... dims) const
	{
		return reshape({ static_cast<size_t>(dims)... });
	}
	
	/// \brief Creates a new tensor with a subview of this tensor's data.
	///
	/// This results in a tensor with one less dimension than this tensor, essentially eliminating the given dimension.
	/// The resulting tensor is not a copy, but a view.
	/// \param dim Which dimension to eliminate.
	/// \param index Which part of the dimension to keep in the resulting tensor.
	/// \return A tensor containing the subview.
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
	
	/// \brief Creates a new constant tensor with a subview of this tensor's data.
	///
	/// This results in a tensor with one less dimension than this tensor, essentially eliminating the given dimension.
	/// The resulting tensor is not a copy, but a view.
	/// \param dim Which dimension to eliminate.
	/// \param index Which part of the dimension to keep in the resulting tensor.
	/// \return A constant tensor containing the subview.
	const Tensor select(size_t dim, size_t index) const
	{
		return const_cast<Tensor *>(this)->select(dim, index);
	}
	
	/// \brief Creates a new tensor with a subview of this tensor's data.
	///
	/// This results in a tensor with the same number of dimensions as this tensor.
	/// The given dimension is narrowed to a specific range, which may be one or more slices in length.
	/// The resulting tensor is not a copy, but a view.
	/// \param dim Which dimension to narrow.
	/// \param index Which part of the dimension to keep in the resulting tensor.
	/// \param size The length of the dimension to keep in the resulting tensor.
	/// \return A tensor containing the subview.
	Tensor narrow(size_t dim, size_t index, size_t size = 1)
	{
		NNAssert(dim < m_dims.size(), "Narrowing dimension out of bounds!");
		NNAssert(index + size <= m_dims[dim], "Out of dimension bounds!");
		Tensor t = *this;
		t.m_offset = m_offset + index * m_strides[dim];
		t.m_dims[dim] = size;
		return t;
	}
	
	/// \brief Creates a new constant tensor with a subview of this tensor's data.
	///
	/// This results in a tensor with the same number of dimensions as this tensor.
	/// The given dimension is narrowed to a specific range, which may be one or more slices in length.
	/// The resulting tensor is not a copy, but a view.
	/// \param dim Which dimension to narrow.
	/// \param index Which part of the dimension to keep in the resulting tensor.
	/// \param size The length of the dimension to keep in the resulting tensor.
	/// \return A constant tensor containing the subview.
	const Tensor narrow(size_t dim, size_t index, size_t size = 1) const
	{
		return const_cast<Tensor *>(this)->narrow(dim, index, size);
	}
	
	/// \brief Makes the given tensor a subview of this tensor's data.
	///
	/// The parameter tensor ends up with the same number of dimensions as this tensor.
	/// This method narrows each dimension according to the values given initializer_list.
	/// Each element in the initializer_list refers to one dimension, in order, and may be
	/// empty (`{}`) for keeping the full dimension,
	/// a single element (i.e. `{3}`) for narrowing to a single slice,
	/// or two elements (i.e. `{3, 2}`) for narrowing a range.
	/// The resulting tensor is not a copy, but a view.
	/// \param t The tensor to use for the subview.
	/// \param dims An initializer_list describing how to narrow each dimension.
	/// \return This tensor, for chaining.
	Tensor &sub(Tensor &t, const std::initializer_list<const std::initializer_list<size_t>> &dims)
	{
		NNAssert(dims.size() == m_dims.size(), "Invalid subtensor dimensions!");
		t.m_offset = m_offset;
		
		size_t dim = 0;
		for(const std::initializer_list<size_t> &params : dims)
		{
			NNAssert(params.size() <= 2, "Invalid parameters for subtensor!");
			if(params.size() == 1)
			{
				size_t index = *params.begin();
				NNAssert(index < m_dims[dim], "Incompatible index!");
				t.m_offset += index * m_strides[dim];
				t.m_dims[dim] = 1;
			}
			else if(params.size() == 2)
			{
				size_t index = *params.begin();
				size_t size = *(params.begin() + 1);
				NNAssert(index + size <= m_dims[dim], "Incompatible index and size!");
				t.m_offset += index * m_strides[dim];
				t.m_dims[dim] = size;
			}
			++dim;
		}
		
		return t;
	}
	
	/// \brief Makes the given tensor a subview of this tensor's data.
	///
	/// The parameter tensor ends up with the same number of dimensions as this tensor.
	/// This method narrows each dimension according to the values given initializer_list.
	/// Each element in the initializer_list refers to one dimension, in order, and may be
	/// empty (`{}`) for keeping the full dimension,
	/// a single element (i.e. `{3}`) for narrowing to a single slice,
	/// or two elements (i.e. `{3, 2}`) for narrowing a range.
	/// The resulting tensor is not a copy, but a view.
	/// \param t The tensor to use for the subview.
	/// \param dims An initializer_list describing how to narrow each dimension.
	/// \return This tensor, for chaining.
	const Tensor &sub(const Tensor &t, const std::initializer_list<const std::initializer_list<size_t>> &dims) const
	{
		return const_cast<Tensor *>(this)->sub(*const_cast<Tensor *>(&t), dims);
	}
	
	/// \brief Creates a new tensor as a subview of this tensor's data.
	///
	/// The new tensor will have the same number of dimensions as this tensor.
	/// This method narrows each dimension according to the values given initializer_list.
	/// Each element in the initializer_list refers to one dimension, in order, and may be
	/// empty (`{}`) for keeping the full dimension,
	/// a single element (i.e. `{3}`) for narrowing to a single slice,
	/// or two elements (i.e. `{3, 2}`) for narrowing a range.
	/// The resulting tensor is not a copy, but a view.
	/// \param dims An initializer_list describing how to narrow each dimension.
	/// \return The narrowed tensor.
	Tensor sub(const std::initializer_list<const std::initializer_list<size_t>> &dims)
	{
		Tensor t = *this;
		return sub(t, dims);
	}
	
	/// \brief Creates a new const tensor as a subview of this tensor's data.
	///
	/// The new tensor will have the same number of dimensions as this tensor.
	/// This method narrows each dimension according to the values given initializer_list.
	/// Each element in the initializer_list refers to one dimension, in order, and may be
	/// empty (`{}`) for keeping the full dimension,
	/// a single element (i.e. `{3}`) for narrowing to a single slice,
	/// or two elements (i.e. `{3, 2}`) for narrowing a range.
	/// The resulting tensor is not a copy, but a view.
	/// \param dims An initializer_list describing how to narrow each dimension.
	/// \return The narrowed const tensor.
	const Tensor sub(const std::initializer_list<const std::initializer_list<size_t>> &dims) const
	{
		return const_cast<Tensor *>(this)->sub(dims);
	}
	
	/// \brief Creates a new tensor as a copy this tensor.
	///
	/// This is a deep copy, and the resulting tensor will have the same shape as this tensor.
	/// \return A copy of this tensor.
	Tensor copy() const
	{
		return reshape(m_dims);
	}
	
	/// \brief Copies the data and shape from another tensor to this tensor.
	///
	/// This is a deep copy, and the tensor will have the same shape as the parameter tensor.
	/// \param other The tensor to copy.
	/// \return This tensor, for chaining.
	Tensor &copy(const Tensor &other)
	{
		NNAssert(size() == other.size(), "Incompatible tensor for copying!");
		auto i = other.begin();
		for(real_t &value : *this)
		{
			value = *i;
			++i;
		}
		return *this;
	}
	
	/// \brief Swaps the data between two tensors.
	///
	/// The given tensor must have the same shape as this tensor.
	/// \param other The tensor with which to swap.
	/// \return This tensor, for chaining.
	Tensor &swap(Tensor &other)
	{
		NNAssert(shape() == other.shape(), "Incompatible tensors for swapping!");
		auto i = other.begin();
		for(real_t &v : *this)
		{
			real_t t = v;
			v = *i;
			*i = t;
			++i;
		}
		return *this;
	}
	
	/// \brief Swaps the data between two tensors.
	///
	/// The given tensor must have the same shape as this tensor.
	/// \param other The tensor with which to swap.
	/// \return This tensor, for chaining.
	/// \note It is alright to use an rvalue reference here, as the temporary tensor is using persistant storage.
	Tensor &swap(Tensor &&other)
	{
		NNAssert(shape() == other.shape(), "Incompatible tensors for swapping!");
		auto i = other.begin();
		for(real_t &v : *this)
		{
			real_t t = v;
			v = *i;
			*i = t;
			++i;
		}
		return *this;
	}
	
	/// \brief Creates a new tensor as a view of this tensor in which two dimensions are switched.
	///
	/// By default, this will switch the first and second dimension, which are rows and columns in a matrix.
	/// The resulting tensor has a view, not a copy, of this tensor.
	/// \param dim1 The first dimension to switch.
	/// \param dim2 The second dimension to switch.
	/// \return A tensor with a subview of this tensor but with the dimensions switched.
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
	
	/// Gets the list of dimension sizes.
	const Storage<size_t> &shape() const
	{
		return m_dims;
	}
	
	/// Gets the number of dimensions in this tensor.
	size_t dims() const
	{
		return m_dims.size();
	}
	
	/// Calculates the total number of elements in this tensor.
	/// \todo Find a faster way; maybe cache the result.
	size_t size() const
	{
		size_t result = 1;
		for(size_t s : m_dims)
		{
			result *= s;
		}
		return result;
	}
	
	/// Gets the size of a given dimension.
	size_t size(size_t dim) const
	{
		NNAssert(dim < m_dims.size(), "Invalid dimension!");
		return m_dims[dim];
	}
	
	/// \brief Gets the stride of a given dimension.
	///
	/// The stide is the distance between elements in the given dimension.
	/// If the dimension is contiguous, the stride in that dimension is 1.
	/// In a completely contiguous tensor, the stride between elements in dimension d
	/// is equal to the stride of dimension d+1 times the size of dimension d+1.
	/// \param dim The dimension from which to get stride.
	/// \return The stride of the given dimension.
	size_t stride(size_t dim) const
	{
		return m_strides[dim];
	}
	
	// MARK: Element manipulation methods.
	
	/// Sets every element in this tensor to the given value.
	Tensor &fill(const real_t &value)
	{
		std::fill(begin(), end(), value);
		return *this;
	}
	
	/// Sets every element in this tensor to 0.
	Tensor &zeros()
	{
		return fill(0);
	}
	
	/// Sets every element in this tensor to 1.
	Tensor &ones()
	{
		return fill(1);
	}
	
	/// \brief Sets every element in this tensor to a uniformly distributed random value.
	///
	/// \param from The lowest value in the uniform distribution.
	/// \param to The highest value in the uniform distribution.
	/// \return This tensor, for chaining.
	Tensor &rand(const real_t &from = -1, const real_t &to = 1)
	{
		for(real_t &v : *this)
		{
			v = Random<real_t>::uniform(from, to);
		}
		return *this;
	}
	
	/// \brief Sets every element in this tensor to a normally distributed random value.
	///
	/// \param mean The mean of the normal distribution.
	/// \param stddev The standard deviation of the normal distribution.
	/// \return This tensor, for chaining.
	Tensor &randn(const real_t &mean = 0, const real_t &stddev = 1)
	{
		for(real_t &v : *this)
		{
			v = Random<real_t>::normal(mean, stddev);
		}
		return *this;
	}
	
	/// \brief Sets every element in this tensor to a normally distributed random value, capped.
	///
	/// This resamples from the distribution when it finds a value too far away from the mean,
	/// which may be slow for a small threshold.
	/// \param mean The mean of the normal distribution.
	/// \param stddev The standard deviation of the normal distribution.
	/// \param cap The threshold value for the maximum allowed distance away from the mean.
	/// \return This tensor, for chaining.
	Tensor &randn(const real_t &mean, const real_t &stddev, const real_t &cap)
	{
		for(real_t &v : *this)
		{
			v = Random<real_t>::normal(mean, stddev, cap);
		}
		return *this;
	}
	
	/// \brief Multiplies this tensor by a scalar.
	///
	/// \param alpha The scalar.
	/// \return This tensor, for chaining.
	Tensor &scale(real_t alpha)
	{
		for(real_t &v : *this)
		{
			v *= alpha;
		}
		return *this;
	}
	
	/// \brief Adds a scalar to each element in this tensor.
	///
	/// \param alpha The scalar.
	/// \return This tensor, for chaining.
	Tensor &shift(real_t alpha)
	{
		for(real_t &v : *this)
		{
			v += alpha;
		}
		return *this;
	}
	
	// MARK: Algebra
	
	/// \todo document this method
	Tensor &multiplyMM(const Tensor &A, const Tensor &B, real_t alpha = 1, real_t beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && stride(1) == 1, "Matrix multiplcation requires contiguous operands!");
		Algebra<real_t>::multiplyMM(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			B.ptr(), B.size(0), B.size(1), B.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMTM(const Tensor &A, const Tensor &B, real_t alpha = 1, real_t beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && stride(1) == 1, "Matrix multiplcation requires contiguous operands!");
		Algebra<real_t>::multiplyMTM(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			B.ptr(), B.size(0), B.size(1), B.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMMT(const Tensor &A, const Tensor &B, real_t alpha = 1, real_t beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && stride(1) == 1, "Matrix multiplcation requires contiguous operands!");
		Algebra<real_t>::multiplyMMT(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			B.ptr(), B.size(0), B.size(1), B.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMTMT(const Tensor &A, const Tensor &B, real_t alpha = 1, real_t beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && stride(1) == 1, "Matrix multiplcation requires contiguous operands!");
		Algebra<real_t>::multiplyMTMT(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			B.ptr(), B.size(0), B.size(1), B.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMV(const Tensor &A, const Tensor &x, real_t alpha = 1, real_t beta = 0)
	{
		NNAssert(A.dims() == 2 && x.dims() == 1 && dims() == 1, "Incompatible operands!");
		NNAssert(A.stride(1) == 1, "Matrix-vector multiplcation requires a contiguous matrix!");
		Algebra<real_t>::multiplyMV(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			x.ptr(), x.size(), x.stride(0),
			ptr(), size(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyMTV(const Tensor &A, const Tensor &x, real_t alpha = 1, real_t beta = 0)
	{
		NNAssert(A.dims() == 2 && x.dims() == 1 && dims() == 1, "Incompatible operands!");
		NNAssert(A.stride(1) == 1, "Matrix-vector multiplcation requires a contiguous matrix!");
		Algebra<real_t>::multiplyMTV(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			x.ptr(), x.size(), x.stride(0),
			ptr(), size(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	Tensor &multiplyVTV(const Tensor &x, const Tensor &y, real_t alpha = 1)
	{
		NNAssert(x.dims() == 1 && y.dims() == 1 && dims() == 2, "Incompatible operands!");
		NNAssert(stride(1) == 1, "Vector outer product requires a contiguous matrix!");
		Algebra<real_t>::multiplyVTV(
			x.ptr(), x.size(), x.stride(0),
			y.ptr(), y.size(), y.stride(0),
			ptr(), size(0), size(1), stride(0),
			alpha
		);
		return *this;
	}
	
	Tensor &addVV(const Tensor &x, real_t alpha = 1)
	{
		NNAssert(x.dims() == 1 && dims() == 1 && x.size() == size(), "Incompatible operands!");
		Algebra<real_t>::addVV(
			x.ptr(), x.size(), x.stride(0),
			ptr(), size(), stride(0),
			alpha
		);
		return *this;
	}
	
	Tensor &addMM(const Tensor &A, real_t alpha = 1)
	{
		NNAssert(A.dims() == 2 && dims() == 2 && A.shape() == shape(), "Incompatible operands!");
		for(size_t i = 0, end = size(0); i < end; ++i)
		{
			Algebra<real_t>::addVV(
				A.ptr() + i * A.stride(0), A.size(1), A.stride(1),
				ptr() + i * stride(0), size(1), stride(1),
				alpha
			);
		}
		return *this;
	}
	
	/// Hadamard/elementwise/pointwise product.
	Tensor &pointwiseProduct(const Tensor &x)
	{
		NNAssert(shape() == x.shape(), "Incompatible operands!");
		auto i = x.begin();
		for(real_t &v : *this)
		{
			v *= *i;
			++i;
		}
		return *this;
	}
	
	// MARK: Statistical methods
	
	real_t sum() const
	{
		real_t result = 0;
		for(const real_t &v : *this)
		{
			result += v;
		}
		return result;
	}
	
	real_t min() const
	{
		real_t result = *begin();
		for(const real_t &v : *this)
		{
			if(v < result)
			{
				result = v;
			}
		}
		return result;
	}
	
	real_t max() const
	{
		real_t result = *begin();
		for(const real_t &v : *this)
		{
			if(v > result)
			{
				result = v;
			}
		}
		return result;
	}
	
	Tensor &normalize(real_t from = 0.0, real_t to = 1.0)
	{
		NNAssert(to > from, "Invalid normalization range!");
		real_t small = min(), large = max();
		return shift(-small).scale((to - from) / (large - small)).shift(from);
	}
	
	// MARK: Element/data access methods.
	
	/// Element access given a multidimensional index.
	real_t &operator()(const Storage<size_t> &indices)
	{
		return (*m_data)[indexOf(indices)];
	}
	
	/// Element access given a multidimensional index.
	const real_t &operator()(const Storage<size_t> &indices) const
	{
		return (*m_data)[indexOf(indices)];
	}
	
	/// Element access given a multidimensional index.
	template <typename ... Ts>
	real_t &operator()(Ts... indices)
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}
	
	/// Element access given a multidimensional index.
	template <typename ... Ts>
	const real_t &operator()(Ts... indices) const
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}
	
	/// Direct raw pointer access.
	real_t *ptr()
	{
		return m_data->ptr() + m_offset;
	}
	
	/// Direct raw pointer access.
	const real_t *ptr() const
	{
		return m_data->ptr() + m_offset;
	}
	
	/// Direct storage access.
	Storage<real_t> &storage()
	{
		return *m_data;
	}
	
	/// Direct storage access.
	const Storage<real_t> &storage() const
	{
		return *m_data;
	}
	
	// MARK: Iterators
	
	TensorIterator<real_t> begin()
	{
		return TensorIterator<real_t>(this);
	}
	
	TensorIterator<real_t> end()
	{
		return TensorIterator<real_t>(this, true);
	}
	
	TensorIterator<const real_t> begin() const
	{
		return TensorIterator<const real_t>(this);
	}
	
	TensorIterator<const real_t> end() const
	{
		return TensorIterator<const real_t>(this, true);
	}
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param out The archive to which to write.
	void save(Archive &out) const
	{
		out << m_dims;
		for(const real_t &x : *this)
			out << x;
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	void load(Archive &in)
	{
		in >> m_dims;
		resize(m_dims);
		for(real_t &x : *this)
			in >> x;
	}
	
private:
	Storage<size_t> m_dims;					///< The length along each dimension.
	Storage<size_t> m_strides;				///< Strides between dimensions.
	size_t m_offset;						///< Offset of data for this view.
	Storage<real_t> *m_data;						///< The actual data.
	std::shared_ptr<Storage<real_t>> m_shared;	///< Wrapped around m_data for ARC.
	
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
TensorIterator<T>::TensorIterator(const Tensor *tensor, bool end) :
	m_tensor(const_cast<Tensor *>(tensor)),
	m_indices(tensor->dims(), 0)
{
	NNAssert(tensor->size() > 0, "Cannot iterate through an empty tensor!");
	if(end)
	{
		m_indices[0] = m_tensor->size(0);
	}
}

template <typename T>
TensorIterator<T> &TensorIterator<T>::advance(size_t dim)
{
	dim = std::min(dim, m_tensor->dims());
	--m_indices.back();
	++m_indices[dim];
	return ++*this;
}

template <typename T>
TensorIterator<T> &TensorIterator<T>::operator++()
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

template <typename T>
TensorIterator<T> TensorIterator<T>::operator++(int)
{
	TensorIterator it = *this;
	return ++it;
}

template <typename T>
T &TensorIterator<T>::operator*()
{
	return (*m_tensor)(m_indices);
}

template <typename T>
bool TensorIterator<T>::operator==(const TensorIterator &other)
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

template <typename T>
bool TensorIterator<T>::operator!=(const TensorIterator &other)
{
	return !(*this == other);
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const Tensor &t)
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
