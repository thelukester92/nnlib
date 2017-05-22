#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <iomanip>
#include <memory>

#include "error.h"
#include "storage.h"
#include "math/math.h"
#include "util/archive.h"
#include "util/random.h"

namespace nnlib
{

template <typename T>
class TensorIterator;

/// \brief The standard input and output type in nnlib.
///
/// A tensor can be a vector (one dimension), a matrix (two dimensions), or a higher-order tensor.
/// Tensors provide views into Storage objects, and multiple tensors can share the same Storage.
template <typename T = double>
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
			for(const T &value : *t)
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
	Tensor(const Storage<T> &values) :
		m_dims({ values.size() }),
		m_strides({ 1 }),
		m_offset(0),
		m_data(new Storage<T>(values)),
		m_shared(m_data)
	{}
	
	/// \brief Create a tensor with the given data.
	///
	/// This performs a deep copy of the data given in the parameter.
	/// The constructed tensor is one-dimensional (a vector).
	/// \param values An initializer_list containing the values to store in the new tensor.
	Tensor(const std::initializer_list<T> &values) :
		m_dims({ values.size() }),
		m_strides({ 1 }),
		m_offset(0),
		m_data(new Storage<T>(values)),
		m_shared(m_data)
	{}
	
	/// \brief Create a tensor with the given shape.
	///
	/// This creates an n-dimensional tensor where n is the size of the input parameter.
	/// \param dims A Storage containing the dimension sizes for the new tensor.
	/// \note This contructor uses a dummy bool to differentiate itself from the const Storage<T> & constructor. This is important when T = size_t.
	Tensor(const Storage<size_t> &dims, bool) :
		m_offset(0),
		m_data(new Storage<T>()),
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
		m_data(new Storage<T>()),
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
	Tensor &operator=(const Storage<T> &values)
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
	Tensor &operator=(const std::initializer_list<T> &values)
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
	/// This will not change the data, although it will result in a contiguous tensor.
	/// Thus, if the tensor was a non-contiguous view, it will end up with a different view.
	/// This method will delete data if resizing smaller or add 0s if resizing bigger.
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
		for(const T &value : *this)
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
	
	/// \brief Creates a new tensor with a "superview" of this tensor's data.
	///
	/// This results in a tensor with the same number of dimensions as this tensor.
	/// The given dimension is expanded from 1 to the given number by repeating the slice n times.
	/// This is accomplished without copying by setting stride to 0 in the given dimension.
	/// The given dimension must already have a size of 1 in the given dimension.
	/// \param dim Which dimension to expand.
	/// \param size How long to expand the given dimension.
	/// \return A tensor containing the "superview."
	Tensor expand(size_t dim, size_t size)
	{
		NNAssert(dim < m_dims.size(), "Expanding dimension out of bounds!");
		NNAssert(m_dims[dim] == 1, "Can only expand a dimension of size 1!");
		Tensor t = *this;
		t.m_dims[dim] = size;
		t.m_strides[dim] = 0;
		return t;
	}
	
	/// \brief Creates a new const tensor with a "superview" of this tensor's data.
	///
	/// This results in a tensor with the same number of dimensions as this tensor.
	/// The given dimension is expanded from 1 to the given number by repeating the slice n times.
	/// This is accomplished without copying by setting stride to 0 in the given dimension.
	/// The given dimension must already have a size of 1 in the given dimension.
	/// \param dim Which dimension to expand.
	/// \param size How long to expand the given dimension.
	/// \return A const tensor containing the "superview."
	const Tensor expand(size_t dim, size_t size) const
	{
		return const_cast<Tensor *>(this)->expand(dim, size);
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
	/// The resulting tensor will also be contiguous, regardless of whether this tensor is contiguous.
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
		for(T &value : *this)
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
		for(T &v : *this)
		{
			T t = v;
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
		for(T &v : *this)
		{
			T t = v;
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
	Tensor &fill(const T &value)
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
	Tensor &rand(const T &from = -1, const T &to = 1)
	{
		for(T &v : *this)
		{
			v = Random<T>::uniform(from, to);
		}
		return *this;
	}
	
	/// \brief Sets every element in this tensor to a normally distributed random value.
	///
	/// \param mean The mean of the normal distribution.
	/// \param stddev The standard deviation of the normal distribution.
	/// \return This tensor, for chaining.
	Tensor &randn(const T &mean = 0, const T &stddev = 1)
	{
		for(T &v : *this)
		{
			v = Random<T>::normal(mean, stddev);
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
	Tensor &randn(const T &mean, const T &stddev, const T &cap)
	{
		for(T &v : *this)
		{
			v = Random<T>::normal(mean, stddev, cap);
		}
		return *this;
	}
	
	/// \brief Multiplies this tensor by a scalar.
	///
	/// \param alpha The scalar.
	/// \return This tensor, for chaining.
	Tensor &scale(T alpha)
	{
		for(T &v : *this)
		{
			v *= alpha;
		}
		return *this;
	}
	
	/// \brief Adds a scalar to each element in this tensor.
	///
	/// \param alpha The scalar.
	/// \return This tensor, for chaining.
	Tensor &add(T alpha)
	{
		for(T &v : *this)
		{
			v += alpha;
		}
		return *this;
	}
	
	// MARK: Algebra
	
	/// Add another vector to this vector.
	Tensor &addV(const Tensor &x, T alpha = 1)
	{
		NNAssert(x.dims() == 1 && dims() == 1 && x.size() == size(), "Incompatible operands!");
		Math<T>::vAdd_v(
			x.ptr(), x.size(), x.stride(0),
			ptr(), stride(0),
			alpha
		);
		return *this;
	}
	
	/// \brief Assigns or adds a matrix/vector with no transposition.
	///
	/// Adds the scaled product of A and x to this vector, scaled.
	/// Sizes must be compatible.
	/// This method will use acceleration, if present.
	/// Effectively, using y for this tensor, this method computes `y = alpha * A * x + beta * y`.
	/// \param A An M x N tensor.
	/// \param x An N tensor.
	/// \param alpha How much to scale A * x.
	/// \param beta How much to scale y.
	/// \return This tensor, for chaining.
	Tensor &assignMV(const Tensor &A, const Tensor &x, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && x.dims() == 1 && dims() == 1, "Incompatible operands!");
		NNAssert(A.stride(1) == 1, "Matrix-vector multiplcation requires a contiguous matrix!");
		Math<T>::vAdd_mv(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			x.ptr(), x.stride(0),
			ptr(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	/// \brief Assigns or adds a matrix/vector with transposition.
	///
	/// Adds the scaled product of A and x to this vector, scaled.
	/// Sizes must be compatible.
	/// This method will use acceleration, if present.
	/// Effectively, using y for this tensor, this method computes `y = alpha * A^T * x + beta * y`.
	/// \param A An N x M tensor.
	/// \param x An N tensor.
	/// \param alpha How much to scale A^T * x.
	/// \param beta How much to scale y.
	/// \return This tensor, for chaining.
	Tensor &assignMTV(const Tensor &A, const Tensor &x, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && x.dims() == 1 && dims() == 1, "Incompatible operands!");
		NNAssert(A.stride(1) == 1, "Matrix-vector multiplcation requires a contiguous matrix!");
		Math<T>::vAdd_mtv(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			x.ptr(), x.stride(0),
			ptr(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	/// \brief Assigns a vector/vector outer product.
	///
	/// Adds the scaled outer product of x and y to this matrix.
	/// Sizes must be compatible.
	/// This method will use acceleration, if present.
	/// Effectively, using A for this tensor, this method computes `A = alpha * x^T * y + A`.
	/// \param x An N tensor.
	/// \param y An M tensor.
	/// \param alpha How much to scale x^T * y.
	/// \return This tensor, for chaining.
	/// \note This method does not include beta because the no-beta version is faster.
	Tensor &assignVV(const Tensor &x, const Tensor &y, T alpha = 1)
	{
		NNAssert(x.dims() == 1 && y.dims() == 1 && dims() == 2, "Incompatible operands!");
		NNAssert(stride(1) == 1, "Vector outer product requires a contiguous matrix!");
		Math<T>::mAdd_vv(
			x.ptr(), x.size(), x.stride(0),
			y.ptr(), y.size(), y.stride(0),
			ptr(), stride(0),
			alpha
		);
		return *this;
	}
	
	/// \brief Assigns or adds a vector/vector outer product.
	///
	/// Adds the scaled outer product of x and y to this matrix.
	/// Sizes must be compatible.
	/// This method will use acceleration, if present.
	/// Effectively, using A for this tensor, this method computes `A = alpha * x^T * y + beta * A`.
	/// \param x An N tensor.
	/// \param y An M tensor.
	/// \param alpha How much to scale x^T * y.
	/// \return This tensor, for chaining.
	/// \note This method does explicitly includes beta because the no-beta version is faster.
	Tensor &assignVV(const Tensor &x, const Tensor &y, T alpha, T beta)
	{
		NNAssert(x.dims() == 1 && y.dims() == 1 && dims() == 2, "Incompatible operands!");
		NNAssert(stride(1) == 1, "Vector outer product requires a contiguous matrix!");
		Math<T>::mAdd_vv(
			x.ptr(), x.size(), x.stride(0),
			y.ptr(), y.size(), y.stride(0),
			ptr(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	/// Add another matrix to this matrix.
	Tensor &addM(const Tensor &A, T alpha = 1)
	{
		NNAssert(A.dims() == 2 && dims() == 2 && A.shape() == shape(), "Incompatible operands!");
		Math<T>::mAdd_m(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			ptr(), stride(0),
			alpha
		);
		return *this;
	}
	
	/// \brief Assigns or adds a matrix multiplcation with no transposition.
	///
	/// Adds the scaled product of A and B to this matrix, scaled.
	/// Sizes must be compatible.
	/// This method will use acceleration, if present.
	/// Effectively, using C for this tensor, this method computes `C = alpha * A * B + beta * C`.
	/// \param A An M x K tensor.
	/// \param B A K x N tensor.
	/// \param alpha How much to scale A * B.
	/// \param beta How much to scale C.
	/// \return This tensor, for chaining.
	Tensor &assignMM(const Tensor &A, const Tensor &B, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && stride(1) == 1, "Matrix multiplcation requires contiguous operands!");
		Math<T>::mAdd_mm(
			A.size(0), B.size(1), A.size(1),
			A.ptr(), A.stride(0),
			B.ptr(), B.stride(0),
			ptr(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	/// \brief Assigns or adds a matrix multiplcation with transposition on the first operand.
	///
	/// Adds the scaled product of A and B to this matrix, scaled.
	/// Sizes must be compatible.
	/// This method will use acceleration, if present.
	/// Effectively, using C for this tensor, this method computes `C = alpha * A^T * B + beta * C`.
	/// \param A A K x M tensor.
	/// \param B A K x N tensor.
	/// \param alpha How much to scale A^T * B.
	/// \param beta How much to scale C.
	/// \return This tensor, for chaining.
	Tensor &assignMTM(const Tensor &A, const Tensor &B, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && stride(1) == 1, "Matrix multiplcation requires contiguous operands!");
		Math<T>::mAdd_mtm(
			A.size(1), B.size(1), A.size(0),
			A.ptr(), A.stride(0),
			B.ptr(), B.stride(0),
			ptr(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	/// \brief Assigns or adds a matrix multiplcation with transposition on the second operand.
	///
	/// Adds the scaled product of A and B to this matrix, scaled.
	/// Sizes must be compatible.
	/// This method will use acceleration, if present.
	/// Effectively, using C for this tensor, this method computes `C = alpha * A * B^T + beta * C`.
	/// \param A An M x K tensor.
	/// \param B An N x K tensor.
	/// \param alpha How much to scale A * B^T.
	/// \param beta How much to scale C.
	/// \return This tensor, for chaining.
	Tensor &assignMMT(const Tensor &A, const Tensor &B, T alpha = 1, T beta = 0)
	{
		NNAssert(A.dims() == 2 && B.dims() == 2 && dims() == 2, "Incompatible operands!");
		NNAssert(A.stride(1) == 1 && B.stride(1) == 1 && stride(1) == 1, "Matrix multiplcation requires contiguous operands!");
		Math<T>::mAdd_mmt(
			A.size(0), B.size(0), A.size(1),
			A.ptr(), A.stride(0),
			B.ptr(), B.stride(0),
			ptr(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	/// Hadamard/elementwise/pointwise product.
	Tensor &pointwiseProduct(const Tensor &x)
	{
		NNAssert(shape() == x.shape(), "Incompatible operands!");
		auto i = x.begin();
		for(T &el : *this)
		{
			el *= *i;
			++i;
		}
		return *this;
	}
	
	/// \brief Compute elementwise/pointwise sum (general purpose).
	///
	/// This is a general purpose function for any size of tensor.
	/// For vectors, addV is called; for matrices, addM is called.
	Tensor &add(const Tensor &x, T alpha = 1)
	{
		NNAssert(shape() == x.shape(), "Incompatible operands to add!");
		if(m_dims.size() == 1)
		{
			return addV(x, alpha);
		}
		else if(m_dims.size() == 2)
		{
			return addM(x, alpha);
		}
		else
		{
			auto i = x.begin();
			for(T &el : *this)
			{
				el += *i;
				++i;
			}
			return *this;
		}
	}
	
	/// Perform a pointwise product with the current tensor, squaring it.
	Tensor &square()
	{
		return pointwiseProduct(*this);
	}
	
	// MARK: Statistical methods
	
	/// Calculate the sum of all elements in this tensor.
	T sum() const
	{
		T result = 0;
		for(const T &v : *this)
		{
			result += v;
		}
		return result;
	}
	
	/// \brief Calculate the sum along the given dimension.
	///
	/// This reduces the number of dimensions by one, and may not be called on a 1D tensor (use sum() instead).
	/// For example, in this matrix:
	/// 	1 2 3
	/// 	4 5 6
	/// sum(0) will produce the vector `<5 7 9>`, and
	/// sum(1) will produce the vector `<6 15>`
	/// \param t The tensor to store the sum in. It must already be the appropriate shape.
	/// \param dim Which dimension to sum.
	/// \return The input tensor t, for chaining.
	Tensor &sum(Tensor &t, size_t dim) const
	{
		NNAssert(dim < m_dims.size(), "Invalid dimension for summation!");
		NNAssert(m_dims.size() > 1, "Cannot sum over a 1D tensor this way! Call sum() instead!");
		
		t.copy(select(dim, 0));
		for(size_t i = 1, n = m_dims[dim]; i < n; ++i)
		{
			t.add(select(dim, i));
		}
		
		return t;
	}
	
	/// \brief Calculate the sum along the given dimension.
	///
	/// This reduces the number of dimensions by one, and may not be called on a 1D tensor (use sum() instead).
	/// For example, in this matrix:
	/// 	1 2 3
	/// 	4 5 6
	/// sum(0) will produce the vector `<5 7 9>`, and
	/// sum(1) will produce the vector `<6 15>`
	/// \param dim Which dimension to sum.
	/// \return The tensor containing the sum.
	Tensor sum(size_t dim) const
	{
		Tensor t(select(dim, 0).shape());
		return sum(t, dim);
	}
	
	T mean() const
	{
		return sum() / size();
	}
	
	T variance() const
	{
		T avg = mean();
		T sum = 0;
		for(const T &v : *this)
		{
			T diff = v - avg;
			sum += diff * diff;
		}
		return sum / size();
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
	
	Tensor &normalize(T from = 0.0, T to = 1.0)
	{
		NNAssert(to > from, "Invalid normalization range!");
		T small = min(), large = max();
		return add(-small).scale((to - from) / (large - small)).add(from);
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
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// The archive takes care of whitespace for plaintext.
	/// \param out The archive to which to write.
	void save(Archive &out) const
	{
		out << m_dims;
		for(const T &x : *this)
			out << x;
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	void load(Archive &in)
	{
		in >> m_dims;
		resize(m_dims);
		for(T &x : *this)
			in >> x;
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
		/// \todo Make the right side of the `||` faster; `size()`` isn't cached, so this will hurt performance.
		if(end || m_tensor->size() == 0)
		{
			m_indices[0] = m_tensor->size(0);
		}
	}
	
	TensorIterator &advance(size_t dim = (size_t) -1)
	{
		dim = std::min(dim, m_tensor->dims());
		--m_indices.back();
		++m_indices[dim];
		return ++*this;
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
	
	TensorIterator operator++(int)
	{
		TensorIterator it = *this;
		return ++it;
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
