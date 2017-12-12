#ifndef CORE_TENSOR_HPP
#define CORE_TENSOR_HPP

#include <iostream>
#include <iomanip>
#include <memory>
#include <functional>

#include "error.hpp"
#include "storage.hpp"
#include "../math/math.hpp"
#include "../util/random.hpp"

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
	/// \brief Vectorizes a list of tensors.
	///
	/// Each tensor in the parameter becomes a subview into a single, contiguous vector.
	/// If the tensors are already vectorized, this returns a vector viewing the entire list.
	/// Otherwise, all shared connections are broken and a new vector is created.
	/// \param tensors A list of tensors to vectorize.
	/// \return The vectorized data (shared by the original tensors).
	static Tensor vectorize(const Storage<Tensor *> &tensors)
	{
		size_t size = 0;
		for(Tensor *t : tensors)
			size += t->size();
		
		if(isVectorized(tensors))
		{
			Tensor flattened = *tensors[0];
			flattened.m_dims = { size };
			flattened.m_strides = { 1 };
			flattened.m_size = size;
			return flattened;
		}
		
		Tensor flattened(size);
		
		size_t offset = 0;
		for(Tensor *t : tensors)
		{
			Tensor view = flattened.narrow(0, offset, t->size());
			view.copy(*t);
			offset += t->size();
			*t = view.view(t->shape());
		}
		
		return flattened;
	}
	
	/// \brief Concatenates a number of tensors along the given dimension.
	///
	/// Each tensor in the parameter becomes a subview into a single shared Storage.
	/// If any of the tensors shared data, the old links are broken and are no longer shared.
	/// Unlike flatten, concatenate requires that the tensors are compatible (same dimensions except for the concatinating dimension).
	/// By default, this will concatenate along the final dimension.
	/// \param tensors A list of tensors to concatenate.
	/// \return The concatenated tensor.
	static Tensor concatenate(const Storage<Tensor *> &tensors, size_t dim = (size_t) -1)
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
		
		Tensor concatenated(dims, true);
		
		// Copy data from tensors into the shared tensor, then give tensors views into the shared tensor
		
		size_t dimOffset = 0;
		for(Tensor *t : tensors)
		{
			Tensor view = concatenated.narrow(dim, dimOffset, t->size(dim));
			view.copy(*t);
			dimOffset += t->size(dim);
			*t = view;
		}
		
		return concatenated;
	}
	
	/// Generate a vector containing a random permutation of integers in [0, n)
	static Tensor randPermutation(size_t n)
	{
		Tensor t(n);
		for(size_t i = 0; i < n; ++i)
			t(i) = i;
		for(size_t i = 1; i < n; ++i)
			std::swap(t(i), t(Random<size_t>::uniform(i + 1)));
		return t;
	}
	
	/// Create a zero-length, one-dimensional tensor.
	Tensor() :
		m_dims({ 0 }),
		m_strides({ 1 }),
		m_offset(0),
		m_data(new Storage<T>()),
		m_shared(m_data),
		m_size(0),
		m_contiguous(true)
	{}
	
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
		m_shared(m_data),
		m_size(values.size()),
		m_contiguous(true)
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
		m_shared(m_data),
		m_size(values.size()),
		m_contiguous(true)
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
	template <typename ... Ts>
	explicit Tensor(size_t dim1, Ts... dims) :
		m_offset(0),
		m_data(new Storage<T>()),
		m_shared(m_data)
	{
		resize({ dim1, static_cast<size_t>(dims)... });
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
		m_shared(other.m_shared),
		m_size(other.m_size),
		m_contiguous(other.m_contiguous)
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
		m_shared(other.m_shared),
		m_size(other.m_size),
		m_contiguous(other.m_contiguous)
	{}
	
	/// Load from a serialized node.
	Tensor(const Serialized &node) : Tensor(node.get<Storage<size_t>>("dims"), true)
	{
		node.get("data", begin(), end());
	}
	
	/// \brief Replace tensor contents with new values.
	///
	/// Resizes the tensor to be a vector (one-dimensional) and copies data from values.
	/// \param values A Storage containing the values to store in the new tensor.
	Tensor &operator=(const Storage<T> &values)
	{
		m_dims			= { values.size() };
		m_strides		= { 1 };
		m_offset		= 0;
		*m_data			= values;
		m_size			= values.size();
		m_contiguous	= true;
		return *this;
	}
	
	/// \brief Replace tensor contents with new values.
	///
	/// Resizes the tensor to be a vector (one-dimensional) and copies data from values.
	/// \param values An initializer_list containing the values to store in the new tensor.
	Tensor &operator=(const std::initializer_list<T> &values)
	{
		m_dims			= { values.size() };
		m_strides		= { 1 };
		m_offset		= 0;
		*m_data			= values;
		m_size			= values.size();
		m_contiguous	= true;
		return *this;
	}
	
	/// \brief Make this tensor a view of another tensor and copy its shape.
	///
	/// This tensor will share the parameter's storage and copy the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	/// \note This essentially performs a shallow copy.
	Tensor &operator=(Tensor &other)
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
	
	/// \brief Move assignment for a tensor.
	///
	/// This tensor will share the parameter's storage and copy the parameter's shape.
	/// \param other The tensor with which to share storage and from which to copy shape.
	/// \note This essentially performs a shallow copy.
	Tensor &operator=(Tensor &&other)
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
	
	// MARK: Size and shape methods.
	
	/// Returns whether this tensor shares a buffer with another tensor.
	bool shared() const
	{
		return m_shared.use_count() > 1;
	}
	
	/// Returns whether this tensor shares a buffer with a specific tensor.
	bool sharedWith(const Tensor &other) const
	{
		return m_data == other.m_data;
	}
	
	/// Returns whether this tensor shares a buffer with all the given tensors.
	bool sharedWith(const Storage<Tensor *> &tensors) const
	{
		for(size_t i = 0, count = tensors.size(); i < count; ++i)
			if(!sharedWith(*tensors[i]))
				return false;
		return true;
	}
	
	/// Returns the number of tensors sharing data with this tensor, including this tensor.
	size_t sharedCount() const
	{
		return m_shared.use_count();
	}
	
	/// \brief Resize this tensor in place and, if necessary, resize its underlying storage.
	///
	/// If this shares data and the new size is greater or this is not contiguous,
	/// this method will break the connction and allocate a new buffer.
	/// When resizing bigger, the new values will be set to 0s.
	/// \param dims The new shape for the tensor.
	///
	/// \return The tensor, for chaining.
	Tensor &resize(const Storage<size_t> &dims)
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
	
	/// \brief Resize this tensor in place and, if necessary, resize its underlying storage.
	///
	/// \param dims A parameter pack containing the new shape for the tensor. Must not be empty.
	/// \return The tensor, for chaining.
	template <typename ... Ts>
	Tensor &resize(size_t dim1, Ts... dims)
	{
		return resize(Storage<size_t>{ dim1, static_cast<size_t>(dims)... });
	}
	
	/// \brief Resize one dimension of this tensor in place and, if necessary, resize its underlying storage.
	///
	/// \param dim Which dimension to resize.
	/// \param size The new size of the given dimension.
	/// \return The tensor, for chaining.
	Tensor &resizeDim(size_t dim, size_t size)
	{
		if(m_dims[dim] == size)
			return *this;
		
		Storage<size_t> dims = m_dims;
		dims[dim] = size;
		
		return resize(dims);
	}
	
	/// \brief Creates a new tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// This must be contiguous and the view must be less than or equal to the current tensor's size.
	/// \param dims A Storage containing the new shape.
	/// \return A tensor that views the same storage as this tensor.
	Tensor view(const Storage<size_t> &dims)
	{
		NNHardAssert(m_contiguous, "Expected a contiguous tensor!");
		
		size_t size = 1;
		for(size_t d : dims)
			size *= d;
		NNHardAssertLessThanOrEquals(size, m_size, "Expected view to be smaller than the original tensor!");
		
		Tensor t = *this;
		return t.resize(dims);
	}
	
	/// \brief Creates a new tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// This must be contiguous and the view must be less than or equal to the current tensor's size.
	/// \param dims A parameter pack containing the new shape.
	/// \return A tensor that views the same storage as this tensor.
	template <typename ... Ts>
	Tensor view(Ts... dims)
	{
		return view({ static_cast<size_t>(dims)... });
	}
	
	/// \brief Creates a new constant tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// This must be contiguous and the view must be less than or equal to the current tensor's size.
	/// \param dims A Storage containing the new shape.
	/// \return A constant tensor that views the same storage as this tensor.
	const Tensor view(const Storage<size_t> &dims) const
	{
		NNHardAssert(m_contiguous, "Expected a contiguous tensor!");
		
		size_t size = 1;
		for(size_t d : dims)
			size *= d;
		NNHardAssertLessThanOrEquals(size, m_size, "Expected view to be smaller than the original tensor!");
		
		Tensor t = *const_cast<Tensor *>(this);
		return t.resize(dims);
	}
	
	/// \brief Creates a new constant tensor with a view of this tensor's storage but (perhaps) a new shape.
	///
	/// This must be contiguous and the view must be less than or equal to the current tensor's size.
	/// \param dims A parameter pack containing the new shape.
	/// \return A constant tensor that views the same storage as this tensor.
	template <typename ... Ts>
	const Tensor view(Ts... dims) const
	{
		Tensor t = const_cast<Tensor *>(this)->view({ static_cast<size_t>(dims)... });
		return t;
	}
	
	/// \brief Creates a new tensor with a copy of this tensor's data and a new shape.
	///
	/// This performs a deep copy of the data.
	/// The given shape must be compatible; that is, the resulting tensor must have as much data as this tensor.
	/// \param dims A Storage containing the new shape.
	/// \return A tensor that with the given shape and a copy of the data in this tensor.
	Tensor reshape(const Storage<size_t> &dims) const
	{
		Tensor t(dims, true);
		NNAssertEquals(t.size(), size(), "Incompatible dimensions for reshaping!");
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
		NNAssertLessThan(dim, m_dims.size(), "Narrowing dimension out of bounds!");
		NNAssertLessThan(index, m_dims[dim], "Out of dimension bounds!");
		Tensor t = *this;
		t.m_offset += index * t.m_strides[dim];
		t.m_dims.erase(dim);
		t.m_strides.erase(dim);
		t.recalculateSize();
		t.checkContiguous();
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
		NNAssertLessThan(dim, m_dims.size(), "Narrowing dimension out of bounds!");
		NNAssertLessThanOrEquals(index + size, m_dims[dim], "Out of dimension bounds!");
		Tensor t = *this;
		t.m_offset = m_offset + index * m_strides[dim];
		t.m_dims[dim] = size;
		t.recalculateSize();
		t.checkContiguous();
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
		NNAssertLessThan(dim, m_dims.size(), "Expanding dimension out of bounds!");
		NNAssertEquals(m_dims[dim], 1, "Can only expand a dimension of size 1!");
		Tensor t = *this;
		t.m_dims[dim] = size;
		t.m_strides[dim] = 0;
		t.recalculateSize();
		t.checkContiguous();
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
	/// This is a deep copy, but the tensor will not necessarily have the same shape, just the same size, as the other tensor.
	/// \param other The tensor to copy.
	/// \return This tensor, for chaining.
	Tensor &copy(const Tensor &other)
	{
		NNAssertEquals(size(), other.size(), "Incompatible tensor for copying!");
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
		NNAssertEquals(shape(), other.shape(), "Incompatible tensors for swapping!");
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
		NNAssertEquals(shape(), other.shape(), "Incompatible tensors for swapping!");
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
		NNAssertLessThan(dim1, dims(), "Invalid dimensions for transposition!");
		NNAssertLessThan(dim2, dims(), "Invalid dimensions for transposition!");
		Tensor t = *this;
		
		size_t temp = t.m_strides[dim1];
		t.m_strides[dim1] = t.m_strides[dim2];
		t.m_strides[dim2] = temp;
		
		temp = t.m_dims[dim1];
		t.m_dims[dim1] = t.m_dims[dim2];
		t.m_dims[dim2] = temp;
		
		t.checkContiguous();
		
		return t;
	}
	
	/// Gets the list of dimension sizes.
	const Storage<size_t> &shape() const
	{
		return m_dims;
	}
	
	/// Gets the list of dimension strides.
	const Storage<size_t> &strides() const
	{
		return m_strides;
	}
	
	/// Gets the number of dimensions in this tensor.
	size_t dims() const
	{
		return m_dims.size();
	}
	
	/// Gets the total number of elements in this tensor.
	size_t size() const
	{
		return m_size;
	}
	
	/// Gets the size of a given dimension.
	size_t size(size_t dim) const
	{
		NNAssertLessThan(dim, m_dims.size(), "Invalid dimension!");
		return m_dims[dim];
	}
	
	/// Gets whether the tensor is contiguous in memory.
	bool contiguous() const
	{
		return m_contiguous;
	}
	
	/// Makes the tensor contiguous in memory.
	Tensor &makeContiguous()
	{
		if(!m_contiguous)
			*this = copy();
		return *this;
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
		for(T &v : *this)
			v = value;
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
			v = Random<T>::uniform(from, to);
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
			v = Random<T>::normal(mean, stddev);
		return *this;
	}
	
	/// \brief Sets every element in this tensor to a value sampled from a Bernoulli distribution (1 or 0).
	///
	/// \param p The probability of a 1.
	/// \return This tensor, for chaining.
	Tensor &bernoulli(const T &p = 0.5)
	{
		for(T &v : *this)
			v = Random<T>::bernoulli(p);
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
			v = Random<T>::normal(mean, stddev, cap);
		return *this;
	}
	
	/// \brief Multiplies this tensor by a scalar.
	///
	/// \param alpha The scalar.
	/// \return This tensor, for chaining.
	Tensor &scale(T alpha)
	{
		for(T &v : *this)
			v *= alpha;
		return *this;
	}
	
	/// \brief Adds a scalar to each element in this tensor.
	///
	/// \param alpha The scalar.
	/// \return This tensor, for chaining.
	Tensor &add(T alpha)
	{
		for(T &v : *this)
			v += alpha;
		return *this;
	}
	
	// MARK: Algebra
	
	/// Add another vector to this vector.
	Tensor &addV(const Tensor &x, T alpha = 1)
	{
		NNAssertEquals(x.dims(), 1, "Expected vector input to addV!");
		NNAssertEquals(dims(), 1, "Expected vector input to addV!");
		NNAssertEquals(x.size(), size(), "Incompatible operands in addV!");
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
		NNAssertEquals(A.dims(), 2, "A must be a matrix!");
		NNAssertEquals(x.dims(), 1, "x must be a vector!");
		NNAssertEquals(dims(), 1, "This must be a vector!");
		NNAssertEquals(x.size(0), A.size(1), "Incompatible operands!");
		NNAssertEquals(size(0), A.size(0), "Incompatible operands!");
		NNAssertEquals(A.stride(1), 1, "A must be contiguous!");
		
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
		NNAssertEquals(A.dims(), 2, "A must be a matrix!");
		NNAssertEquals(x.dims(), 1, "x must be a vector!");
		NNAssertEquals(dims(), 1, "This must be a vector!");
		NNAssertEquals(x.size(0), A.size(0), "Incompatible operands!");
		NNAssertEquals(size(0), A.size(1), "Incompatible operands!");
		NNAssertEquals(A.stride(1), 1, "A must be contiguous!");
		
		Math<T>::vAdd_mtv(
			A.ptr(), A.size(0), A.size(1), A.stride(0),
			x.ptr(), x.stride(0),
			ptr(), stride(0),
			alpha, beta
		);
		return *this;
	}
	
	/// \brief Assigns or adds a vector/vector outer product.
	///
	/// Adds the scaled outer product of x and y to this matrix.
	/// Sizes must be compatible.
	/// This method will use acceleration, if present.
	/// Effectively, using A for this tensor, this method computes `A = alpha * x^T * y + A`.
	/// \param x An N tensor.
	/// \param y An M tensor.
	/// \param alpha How much to scale x^T * y.
	/// \return This tensor, for chaining.
	Tensor &assignVV(const Tensor &x, const Tensor &y, T alpha = 1, T beta = 0)
	{
		NNAssertEquals(x.dims(), 1, "x must be a vector!");
		NNAssertEquals(y.dims(), 1, "y must be a vector!");
		NNAssertEquals(dims(), 2, "This must be a matrix!");
		NNAssertEquals(size(0), x.size(0), "Incompatible operands!");
		NNAssertEquals(size(1), y.size(0), "Incompatible operands!");
		NNAssertEquals(stride(1), 1, "This must be contiguous!");
		
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
		NNAssertEquals(A.dims(), 2, "A must be a matrix!");
		NNAssertEquals(dims(), 2, "This must be a matrix!");
		NNAssertEquals(A.shape(), shape(), "Incompatible operands!");
		
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
		NNAssertEquals(A.dims(), 2, "A must be a matrix!");
		NNAssertEquals(B.dims(), 2, "B must be a matrix!");
		NNAssertEquals(dims(), 2, "This must be a matrix!");
		NNAssertEquals(A.stride(1), 1, "A must be contiguous!");
		NNAssertEquals(B.stride(1), 1, "B must be contiguous!");
		NNAssertEquals(stride(1), 1, "This must be contiguous!");
		NNAssertEquals(A.size(0), size(0), "Incompatible operands!");
		NNAssertEquals(B.size(1), size(1), "Incompatible operands!");
		NNAssertEquals(A.size(1), B.size(0), "Incompatible operands!");
		
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
		NNAssertEquals(A.dims(), 2, "A must be a matrix!");
		NNAssertEquals(B.dims(), 2, "B must be a matrix!");
		NNAssertEquals(dims(), 2, "This must be a matrix!");
		NNAssertEquals(A.stride(1), 1, "A must be contiguous!");
		NNAssertEquals(B.stride(1), 1, "B must be contiguous!");
		NNAssertEquals(stride(1), 1, "This must be contiguous!");
		NNAssertEquals(A.size(1), size(0), "Incompatible operands!");
		NNAssertEquals(B.size(1), size(1), "Incompatible operands!");
		NNAssertEquals(A.size(0), B.size(0), "Incompatible operands!");
		
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
		NNAssertEquals(A.dims(), 2, "A must be a matrix!");
		NNAssertEquals(B.dims(), 2, "B must be a matrix!");
		NNAssertEquals(dims(), 2, "This must be a matrix!");
		NNAssertEquals(A.stride(1), 1, "A must be contiguous!");
		NNAssertEquals(B.stride(1), 1, "B must be contiguous!");
		NNAssertEquals(stride(1), 1, "This must be contiguous!");
		NNAssertEquals(A.size(0), size(0), "Incompatible operands!");
		NNAssertEquals(B.size(0), size(1), "Incompatible operands!");
		NNAssertEquals(A.size(1), B.size(1), "Incompatible operands!");
		
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
		NNAssertEquals(shape(), x.shape(), "Incompatible operands!");
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
		NNAssertEquals(shape(), x.shape(), "Incompatible operands to add!");
		if(m_dims.size() == 1)
			return addV(x, alpha);
		else if(m_dims.size() == 2)
			return addM(x, alpha);
		else
		{
			auto i = x.begin();
			for(T &el : *this)
			{
				el += *i * alpha;
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
	
	/// \brief Sparsify the current dense tensor, dropping values with magnitude less than epsilon.
	///
	/// The output will be a matrix. The number of rows will be the number of non-zero elements;
	/// the number of columns will be D + 1 where D is the number of dimensions in the dense tensor.
	/// The first D columns in a row are the indices and the last is the value in that slot.
	/// The first row in the output will be the sizes of each dimension with one unused column.
	///
	/// For example, a truncated identity matrix of size 3x5 could be represented like this:
	///
	///     3 5 0.0   <-- size
	///     0 0 1.0
	///     1 1 1.0
	///     2 2 1.0
	Tensor sparsify(T epsilon = 1e-12)
	{
		size_t count = 0;
		for(auto x : *this)
			if(std::abs(x) > epsilon)
				++count;
		
		Tensor<T> sparse(count, m_dims.size() + 1);
		
		size_t idx = 0;
		for(auto i = begin(), iend = end(); i != iend; ++i)
		{
			if(std::abs(*i) > epsilon)
			{
				for(size_t j = 0, jend = i.indices().size(); j != jend; ++j)
					sparse(idx, j) = i.indices()(j);
				sparse(idx, i.indices().size()) = *i;
				++idx;
			}
		}
		
		return sparse;
	}

	/// \brief Unsparsify the current sparse tensor.
	///
	/// See sparsify for an explanation of sparse tensors.
	Tensor unsparsify()
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
	
	// MARK: Functional
	
	/// \brief Apply the given function to each element in this tensor.
	///
	/// \note We may eventually split to apply(V|M) (see the add method) for acceleration.
	Tensor &apply(const std::function<void(T&)> &f)
	{
		for(T &val : *this)
			f(val);
		return *this;
	}
	
	/// \brief Apply the given function to each element in this tensor.
	///
	/// \note We may eventually split to apply(V|M) (see the add method) for acceleration.
	const Tensor &apply(const std::function<void(const T&)> &f) const
	{
		for(const T &val : *this)
			f(val);
		return *this;
	}
	
	// MARK: Statistical methods
	
	/// Calculate the sum of all elements in this tensor.
	T sum() const
	{
		T result = 0;
		for(const T &v : *this)
			result += v;
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
		NNAssertLessThan(dim, m_dims.size(), "Invalid dimension for summation!");
		NNAssertGreaterThan(m_dims.size(), 1, "Cannot sum over a 1D tensor this way! Call sum() instead!");
		
		t.copy(select(dim, 0));
		for(size_t i = 1, n = m_dims[dim]; i < n; ++i)
			t.add(select(dim, i));
		
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
		Tensor t(select(dim, 0).shape(), true);
		return sum(t, dim);
	}
	
	/// Calculate the mean of the elements of this tensor.
	T mean() const
	{
		return sum() / size();
	}
	
	/// Calculate the variance of the elements of this tensor.
	T variance(bool normalizeAsSample = false) const
	{
		T avg = mean();
		T sum = 0;
		for(const T &v : *this)
		{
			T diff = v - avg;
			sum += diff * diff;
		}
		return sum / (size() + (normalizeAsSample ? 1 : 0));
	}
	
	/// Find the minimum element of this tensor.
	T min() const
	{
		T result = *ptr();
		for(const T &v : *this)
			if(v < result)
				result = v;
		return result;
	}
	
	/// Find the maximum element of this tensor.
	T max() const
	{
		T result = *ptr();
		for(const T &v : *this)
			if(v > result)
				result = v;
		return result;
	}
	
	/// Normalize the elements of this tensor.
	Tensor &normalize(T from = 0.0, T to = 1.0)
	{
		NNAssertGreaterThan(to, from, "Invalid normalization range!");
		T small = min(), large = max();
		return add(-small).scale((to - from) / (large - small)).add(from);
	}
	
	/// Clip the elements of this tensor such that all elements lie in [smallest, largest]
	Tensor &clip(T smallest, T largest)
	{
		NNAssertGreaterThan(largest, smallest, "Invalid clipping range!");
		for(T &v : *this)
			v = std::min(std::max(v, smallest), largest);
		return *this;
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
	
	/// Save to a serialized node.
	void save(Serialized &node) const
	{
		node.set("dims", m_dims);
		node.set("data", begin(), end());
	}
	
private:
	Storage<size_t> m_dims;					///< The length along each dimension.
	Storage<size_t> m_strides;				///< Strides between dimensions.
	size_t m_offset;						///< Offset of data for this view.
	Storage<T> *m_data;						///< The actual data.
	std::shared_ptr<Storage<T>> m_shared;	///< Wrapped around m_data for ARC.
	size_t m_size;							///< The total number of elements.
	bool m_contiguous;						///< Whether this tensor is contiguous (i.e. can be vectorized).
	
	/// Check whether the given list of tensors is already vectorized.
	static bool isVectorized(const Storage<Tensor *> &tensors)
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
	
	/// Get the appropriate contiguous index given the multidimensional index.
	size_t indexOf(const std::initializer_list<size_t> &indices) const
	{
		NNAssertEquals(indices.size(), m_dims.size(), "Incorrect number of dimensions!");
		size_t i = 0, sum = m_offset;
		for(size_t idx : indices)
			sum += idx * m_strides[i++];
		
		NNAssertLessThan(sum, m_data->size(), "Index out of bounds!");
		return sum;
	}
	
	/// Get the appropriate contiguous index given the multidimensional index.
	size_t indexOf(const Storage<size_t> &indices) const
	{
		NNAssertEquals(indices.size(), m_dims.size(), "Incorrect number of dimensions!");
		size_t sum = m_offset;
		for(size_t i = 0, j = indices.size(); i < j; ++i)
			sum += indices[i] * m_strides[i];
		
		NNAssertLessThan(sum, m_data->size(), "Index out of bounds!");
		return sum;
	}
	
	/// Recalculate and cache the size of this tensor.
	void recalculateSize()
	{
		m_size = 1;
		for(size_t s : m_dims)
			m_size *= s;
	}
	
	/// Check and cache whether this tensor is contiguous.
	void checkContiguous()
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
};

template <typename T>
class TensorIterator : public std::iterator<std::forward_iterator_tag, T, std::ptrdiff_t, const T *, T &>
{
using TT = typename std::remove_const<T>::type;
public:
	TensorIterator(const Tensor<TT> *tensor, bool end = false) :
		m_contiguous(tensor->contiguous()),
		m_shape(tensor->shape()),
		m_stride(tensor->strides()),
		m_indices(m_contiguous ? 1 : tensor->dims(), 0),
		m_ptr(const_cast<Tensor<TT> *>(tensor)->ptr())
	{
		if(end || tensor->size() == 0)
		{
			m_indices[0] = m_shape[0];
			m_ptr += m_stride[0] * m_indices[0];
		}
	}
	
	TensorIterator &operator++()
	{
		if(m_contiguous)
		{
			++m_ptr;
			return *this;
		}
		
		size_t d = m_indices.size() - 1;
		++m_indices[d];
		m_ptr += m_stride[d];
		
		while(m_indices[d] >= m_shape[d] && d > 0)
		{
			m_ptr -= m_stride[d] * m_indices[d];
			m_indices[d] = 0;
			
			--d;
			
			++m_indices[d];
			m_ptr += m_stride[d];
		}
		
		return *this;
	}
	
	TensorIterator operator++(int)
	{
		TensorIterator it = *this;
		++(*this);
		return it;
	}
	
	T &operator*()
	{
		return *m_ptr;
	}
	
	bool operator==(const TensorIterator &other)
	{
		return !(*this != other);
	}
	
	bool operator!=(const TensorIterator &other)
	{
		if(m_contiguous)
			return m_ptr != other.m_ptr;
		else
			return m_indices != other.m_indices;
	}
	
private:
	bool m_contiguous;
	const Storage<size_t> &m_shape;
	const Storage<size_t> &m_stride;
	Storage<size_t> m_indices;
	TT *m_ptr;
};

}

#endif
