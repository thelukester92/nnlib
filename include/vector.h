#ifndef VECTOR_H
#define VECTOR_H

#include <algorithm>
#include <memory>
#include <iterator>
#include "error.h"

namespace nnlib
{

// MARK: Interface

template <typename T>
class Vector
{
public:
	class Iterator : public std::iterator<std::forward_iterator_tag, T>
	{
	public:
		Iterator(T *ptr, size_t stride) : m_ptr(ptr), m_stride(stride) {}
		Iterator &operator++()				{ m_ptr += m_stride; return *this; }
		T &operator*()						{ return *m_ptr; }
		bool operator==(const Iterator &i)	{ return m_ptr == i.m_ptr && m_stride == i.m_stride; }
		bool operator!=(const Iterator &i)	{ return m_ptr != i.m_ptr || m_stride != i.m_stride; }
	private:
		T *m_ptr;
		size_t m_stride;
	};
	
	Vector(size_t n = 0);
	void fill(const T &val);
	
	T &operator[](size_t i);
	const T&operator[](size_t i) const;
	T &operator()(size_t i);
	const T &operator()(size_t i) const;
	
	Iterator begin();
	Iterator end();
private:
	T *m_ptr;
	size_t m_stride;
	size_t m_size;
	size_t m_capacity;
	
	std::shared_ptr<T> m_shared;
};

// MARK: Implementation

template <typename T>
Vector<T>::Vector(size_t n) : m_ptr(new T[n]), m_stride(1), m_size(n), m_capacity(n), m_shared(m_ptr)
{
	fill(T());
}

template <typename T>
void Vector<T>::fill(const T &val)
{
	std::fill(begin(), end(), val);
}

template <typename T>
typename Vector<T>::Iterator Vector<T>::begin()
{
	return Iterator(m_ptr, m_stride);
}

template <typename T>
typename Vector<T>::Iterator Vector<T>::end()
{
	return Iterator(m_ptr + m_stride * m_size, m_stride);
}

template <typename T>
T &Vector<T>::operator[](size_t i)
{
	NNAssert(i < m_size, "Invalid Vector index!");
	return m_ptr[i * m_stride];
}

template <typename T>
const T &Vector<T>::operator[](size_t i) const
{
	NNAssert(i < m_size, "Invalid Vector index!");
	return m_ptr[i * m_stride];
}

template <typename T>
T &Vector<T>::operator()(size_t i)
{
	NNAssert(i < m_size, "Invalid Vector index!");
	return m_ptr[i * m_stride];
}

template <typename T>
const T &Vector<T>::operator()(size_t i) const
{
	NNAssert(i < m_size, "Invalid Vector index!");
	return m_ptr[i * m_stride];
}

}

#endif
