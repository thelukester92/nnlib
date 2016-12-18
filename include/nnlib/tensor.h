#ifndef TENSOR_H
#define TENSOR_H

#include <type_traits>
#include <memory>
#include "error.h"

namespace nnlib
{

/// A base tensor class from which Vector and Matrix derive.
/// Provides basic storage; Vector/Matrix provide access.
template <typename T>
class Tensor
{
friend class Tensor<const T>;
friend class Tensor<typename std::remove_const<T>::type>;
public:
	Tensor(size_t n = 0) : m_ptr(new T[n]), m_size(n), m_shared(m_ptr) {}
	Tensor(const Tensor &t) : m_ptr(t.m_ptr), m_size(t.m_size), m_shared(t.m_shared) {}
	Tensor(const Tensor &t, size_t offset, size_t size) : m_ptr(t.m_ptr + offset), m_size(size), m_shared(t.m_shared) {}
	
	/// Create a const version of a non-const tensor.
	template <typename U = T>
	Tensor(const Tensor<typename std::enable_if<std::is_const<U>::value, typename std::remove_const<U>::type>::type> &t) : m_ptr(t.m_ptr), m_size(t.m_size), m_shared(t.m_shared) {}
	
	/// The raw underlying storage.
	T *ptr()
	{
		return m_ptr;
	}
	
	/// The number of elements in this tensor.
	size_t size() const
	{
		return m_size;
	}
	
	/// Reallocate this tensor; this assumes contiguous storage and may decouple shared tensors.
	Tensor &resize(size_t size)
	{
		if(size > m_size)
		{
			T *ptr = new T[size];
			for(size_t i = 0; i < m_size; ++i)
				ptr[i] = m_ptr[i];
			m_size = size;
			m_shared.reset(m_ptr = ptr);
		}
		return *this;
	}
	
	/// Change the underlying pointer.
	Tensor &set(T *ptr, size_t size, const std::shared_ptr<T> &shared)
	{
		m_ptr		= ptr;
		m_size		= size;
		m_shared	= shared;
		return *this;
	}
protected:
	T *m_ptr;						///< The primary storage, with offset.
	size_t m_size;					///< The number of elements.
	std::shared_ptr<T> m_shared;	///< For ARC.
};

}

#endif
