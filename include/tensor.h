#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <memory>
#include <iterator>
#include "error.h"

namespace nnlib
{

/// A base tensor class from which Vector and Matrix derive.
/// Provides basic storage; Vector/Matrix provide access.
template <typename T>
class Tensor
{
public:
	Tensor(size_t n = 0) : m_ptr(new T[n]), m_size(n), m_shared(m_ptr) {}
	Tensor(const Tensor &t, size_t offset, size_t size) : m_ptr(t.m_ptr + offset), m_size(size), m_shared(t.m_shared) {}
protected:
	T *m_ptr;						///< The primary storage, with offset.
	size_t m_size;					///< The number of elements.
	std::shared_ptr<T> m_shared;	///< For ARC.
};

}

#endif
