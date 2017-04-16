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
	
	// MARK: Size and shape methods.
	
	/// Resize this tensor and, if necessary, its underlying storage.
	template <typename ... Ts>
	void resize(Ts... dims)
	{
		m_dims = { static_cast<size_t>(dims)... };
		m_strides.resize(m_dims.size());
		
		m_strides[m_strides.size() - 1] = 1;
		for(size_t i = m_strides.size() - 1; i > 0; --i)
		{
			m_strides[i - 1] = m_strides[i] * m_dims[i];
		}
		
		m_data.resize(m_strides[0] * m_dims[0]);
	}
	
	/// Reshape this tensor without changing its size.
	template <typename ... Ts>
	void reshape(Ts... dims)
	{
		Storage<size_t> newDims = { static_cast<size_t>(dims)... };
		Storage<size_t> newStrides(newDims.size());
		
		newStrides[newStrides.size() - 1] = 1;
		for(size_t i = newStrides.size() - 1; i > 0; --i)
		{
			newStrides[i - 1] = newStrides[i] * newDims[i];
		}
		
		NNAssert(newStrides[0] * newDims[0] == m_data.size(), "Incompatible shape!");
		m_data.resize(newStrides[0] * newDims[0]);
		
		m_dims = newDims;
		m_strides = newStrides;
	}
	
	/// Get the number of dimensions in this tensor.
	size_t dims() const
	{
		return m_dims.size();
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
		return m_data[indexOf({ static_cast<size_t>(indices)... })];
	}
private:
	Storage<size_t> m_dims, m_strides;
	Storage<T> m_data;
	
	/// Get the appropriate contiguous index given the multidimensional index.
	/// \todo BLAS accelerated dot product between indices and m_strides
	size_t indexOf(const std::initializer_list<size_t> &indices)
	{
		NNAssert(indices.size() == m_dims.size(), "Incorrect number of dimensions!");
		size_t sum = 0, idx = 0;
		for(const size_t *i = indices.begin(), *j = indices.end(); i != j; ++i, ++idx)
			sum += *i * m_strides[idx];
		std::cout << sum << std::endl;
		return sum;
	}
};

}

#endif
