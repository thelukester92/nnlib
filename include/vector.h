#ifndef VECTOR_H
#define VECTOR_H

#include <type_traits>
#include <iterator>
#include <algorithm>
#include "tensor.h"

namespace nnlib
{

/// Standard 1-dimensional tensor.
template <typename T>
class Vector : public Tensor<T>
{
friend class Vector<const T>;
friend class Vector<typename std::remove_const<T>::type>;
using Tensor<T>::m_ptr;
using Tensor<T>::m_size;
public:
	class Iterator : public std::iterator<std::forward_iterator_tag, T>
	{
	public:
		Iterator(T *ptr, size_t stride)		: m_ptr(ptr), m_stride(stride) {}
		Iterator &operator++()				{ m_ptr += m_stride; return *this; }
		T &operator*()						{ return *m_ptr; }
		bool operator==(const Iterator &i)	{ return m_ptr == i.m_ptr && m_stride == i.m_stride; }
		bool operator!=(const Iterator &i)	{ return m_ptr != i.m_ptr || m_stride != i.m_stride; }
	private:
		T *m_ptr;
		size_t m_stride;
	};
	
	// MARK: Constructors
	
	/// Create a vector of size n.
	Vector(size_t n = 0) : Tensor<T>(n), m_stride(1)
	{
		fill(T());
	}
	
	/// Create a shallow copy of another vector.
	Vector(const Vector &v) : Tensor<T>(v), m_stride(v.m_stride) {}
	
	/// Create a shallow copy of another tensor (i.e. matrix).
	Vector(const Tensor<T> &t, size_t offset, size_t size, size_t stride = 1) : Tensor<T>(t, offset, size), m_stride(stride) {}
	
	/// Create a const view of a non-const vector.
	template <typename U = T>
	Vector(const Vector<typename std::enable_if<std::is_const<U>::value, typename std::remove_const<U>::type>::type> &v) : Tensor<T>(v), m_stride(v.m_stride) {}
	
	// MARK: Element Manipulation
	
	void fill(const T &val)
	{
		std::fill(begin(), end(), val);
	}
	
	// MARK: Element Access
	
	T &operator[](size_t i)
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		return m_ptr[i * m_stride];
	}
	
	const T &operator[](size_t i) const
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		return m_ptr[i * m_stride];
	}
	
	T &operator()(size_t i)
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		return m_ptr[i * m_stride];
	}
	
	const T &operator()(size_t i) const
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		return m_ptr[i * m_stride];
	}
	
	// MARK: Iterators
	
	Iterator begin()
	{
		return Iterator(m_ptr, m_stride);
	}
	
	Iterator end()
	{
		return Iterator(m_ptr + m_stride * m_size, m_stride);
	}
private:
	size_t m_stride;	///< The stride between elements in this Vector
};

/// Constant 1-dimensional view of a tensor.
template <typename T>
using ConstVector = Vector<const T>;

}

#endif
