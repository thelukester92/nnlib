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
protected:
	T *m_ptr;						///< The primary storage, with offset.
	size_t m_size;					///< The number of elements.
	std::shared_ptr<T> m_shared;	///< For ARC.
};

}

#endif
