#ifndef TENSOR_H
#define TENSOR_H

#include <memory>

#include "error.h"
#include "storage.h"
#include "algebra.h"

namespace nnlib
{

template <typename T>
class Tensor
{
public:
	/// Create a tensor with the given size and shape.
	template <typename ... Ts>
	Tensor(Ts... dims) :
		m_data(new Storage<T>()),
		m_shared(m_data)
	{
		resize(dims...);
	}
	
	explicit Tensor(const Storage<size_t> &dims) :
		m_data(new Storage<T>()),
		m_shared(m_data)
	{
		resize(dims);
	}
	
	/// Create a tensor as a view of another tensor with the same shape.
	/// \note Non-rvalue tensors can use default shallow copy.
	Tensor(Tensor &&other) :
		m_dims(other.m_dims),
		m_strides(other.m_strides),
		m_data(other.m_data),
		m_shared(other.m_shared)
	{}
	
	// MARK: Size and shape methods.
	
	/// Resize this tensor in place and, if necessary, resizes its underlying storage.
	Tensor &resize(const Storage<size_t> &dims)
	{
		m_dims = dims;
		m_strides.resize(m_dims.size());
		
		m_strides[m_strides.size() - 1] = 1;
		for(size_t i = m_strides.size() - 1; i > 0; --i)
		{
			m_strides[i - 1] = m_strides[i] * m_dims[i];
		}
		
		m_data->resize(m_strides[0] * m_dims[0]);
		return *this;
	}
	
	template <typename ... Ts>
	Tensor &resize(Ts... dims)
	{
		return resize({ static_cast<size_t>(dims)... });
	}
	
	/// Creates a new tensor with a copy of this data but a new shape.
	/// The shape must be compatible.
	Tensor reshape(const Storage<size_t> &dims)
	{
		Tensor t(dims);
		NNAssert(t.size() == m_data->size(), "Incompatible dimensions for reshaping!");
		/// \todo copy all data into t
		return t;
	}
	
	template <typename ... Ts>
	Tensor reshape(Ts... dims)
	{
		return reshape({ static_cast<size_t>(dims)... });
	}
	
	/// Get the number of dimensions in this tensor.
	size_t dims() const
	{
		return m_dims.size();
	}
	
	/// Get the total number of elements in this tensor.
	size_t size() const
	{
		return m_data->size();
	}
	
	/// Get the size of a given dimension.
	size_t size(size_t dim) const
	{
		return m_dims[dim];
	}
	
	/// Get the stride of a given dimension.
	size_t stride(size_t dim) const
	{
		return m_strides[dim];
	}
	
	// MARK: Element manipulation methods.
	
	Tensor &fill(const T &value)
	{
		/// \todo fill me in; may need to implement iterators first and/or isContigious
	}
	
	// MARK: Element access methods.
	
	/// Element access given a multidimensional index.
	template <typename ... Ts>
	T &operator()(Ts... indices)
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}
private:
	Storage<size_t> m_dims;					///< The length along each dimension.
	Storage<size_t> m_strides;				///< Strides between dimensions.
	Storage<T> *m_data;						///< The actual data.
	std::shared_ptr<Storage<T>> m_shared;	///< Wrapped around m_data for ARC.
	
	/// Get the appropriate contiguous index given the multidimensional index.
	size_t indexOf(const Storage<size_t> &indices)
	{
		NNAssert(indices.size() == m_dims.size(), "Incorrect number of dimensions!");
		return Algebra<size_t>::dot(indices.size(), indices.begin(), 1, m_dims.ptr(), 1);
	}
};

template <typename T>
class TensorIterator
{
public:
	TensorIterator &operator++();
	TensorIterator operator++(int);
};

}

#endif
