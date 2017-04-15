#ifndef TENSOR_H
#define TENSOR_H

namespace nnlib
{

#include "error.h"
#include "storage.h"

template <typename T>
class Tensor
{
public:
	/// Create a tensor with the given size and shape.
	template <typename ... Ts>
	Tensor(Ts... dims)
	{
		resize(dims...);
	}
	
	/// Resize this tensor and, if necessary, its underlying storage.
	template <typename ... Ts>
	void resize(Ts... dims)
	{
		m_dims = { static_cast<size_t>(dims)... };
		size_t product = 1;
		for(size_t i = 0, j = m_dims.size(); i < j; ++i)
		{
			product *= m_dims[i];
		}
		m_data.resize(product);
	}
	
	/// Reshape this tensor without changing its size.
	template <typename ... Ts>
	void reshape(Ts... dims)
	{
		Storage<size_t> newDims = { static_cast<size_t>(dims)... };
		size_t product = 1;
		for(size_t i = 0, j = m_dims.size(); i < j; ++i)
		{
			product *= newDims[i];
		}
		NNAssert(product == m_data.size(), "Incompatible shape!");
		m_dims = newDims;
	}
	
	/// Get the total number of elements in this tensor.
	size_t size() const
	{
		return m_data.size();
	}
	
	/// Get the size of a given dimension.
	size_t size(size_t dim) const
	{
		return m_dims[dim];
	}
	
	/// Get the number of dimensions in this tensor.
	size_t dims() const
	{
		return m_dims.size();
	}
	
	/// Element access given a multidimensional index.
	template <typename ... Ts>
	T &operator()(Ts... indices)
	{
		return m_data[indexOf({ static_cast<size_t>(indices)... })];
	}
private:
	Storage<size_t> m_dims;
	Storage<T> m_data;
	
	/// Get the appropriate contiguous index given the multidimensional index.
	size_t indexOf(const std::initializer_list<size_t> &indices)
	{
		NNAssert(indices.size() == m_dims.size(), "Incorrect number of dimensions!");
		size_t result = 0, dim = 1;
		const size_t *i, *j;
		for(i = indices.begin(), j = indices.end() - 1; i != j; ++i, ++dim)
		{
			result += *i;
			result *= m_dims[dim];
		}
		return result + *i;
	}
};

}

#endif
