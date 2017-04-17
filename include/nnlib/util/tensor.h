#ifndef TENSOR_H
#define TENSOR_H

#include <memory>

#include "error.h"
#include "storage.h"
#include "algebra.h"

namespace nnlib
{

template <typename T>
class TensorIterator;

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
	
	Tensor &operator=(Tensor &&other)
	{
		m_dims		= other.m_dims;
		m_strides	= other.m_strides;
		m_data		= other.m_data;
		m_shared	= other.m_shared;
	}
	
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
	Tensor reshape(const Storage<size_t> &dims) const
	{
		Tensor t(dims);
		NNAssert(t.size() == m_data->size(), "Incompatible dimensions for reshaping!");
		auto k = t.begin();
		for(const T &value : *this)
		{
			*k = value;
			++k;
		}
		return t;
	}
	
	template <typename ... Ts>
	Tensor reshape(Ts... dims) const
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
		NNAssert(dim < m_dims.size(), "Invalid dimension!");
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
		for(T &v : *this)
		{
			v = value;
		}
		return *this;
	}
	
	// MARK: Element access methods.
	
	/// Element access given a multidimensional index.
	T &operator()(const Storage<size_t> &indices)
	{
		return (*m_data)[indexOf(indices)];
	}
	
	template <typename ... Ts>
	T &operator()(Ts... indices)
	{
		return (*m_data)[indexOf({ static_cast<size_t>(indices)... })];
	}
	
	// MARK: iterators
	
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
	Storage<T> *m_data;						///< The actual data.
	std::shared_ptr<Storage<T>> m_shared;	///< Wrapped around m_data for ARC.
	
	/// Get the appropriate contiguous index given the multidimensional index.
	size_t indexOf(const Storage<size_t> &indices)
	{
		NNAssert(indices.size() == m_dims.size(), "Incorrect number of dimensions!");
		size_t sum = 0;
		for(size_t i = 0, j = indices.size(); i < j; ++i)
		{
			sum += indices[i] * m_strides[i];
		}
		return sum;
	}
};

template <typename T>
class TensorIterator
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

}

#endif
