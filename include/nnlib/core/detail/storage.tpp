#ifndef CORE_STORAGE_TPP
#define CORE_STORAGE_TPP

#include "../storage.hpp"

namespace nnlib
{

template <typename T>
Storage<T>::Storage(size_t n, const T &defaultValue) :
	m_ptr(new T[n]),
	m_size(n),
	m_capacity(n)
{
	for(size_t i = 0; i < n; ++i)
		m_ptr[i] = defaultValue;
}

template <typename T>
Storage<T>::Storage(const Storage<T> &copy) :
	m_ptr(new T[copy.size()]),
	m_size(copy.size()),
	m_capacity(copy.size())
{
	size_t index = 0;
	for(const T &value : copy)
	{
		m_ptr[index] = value;
		++index;
	}
}

template <typename T>
Storage<T>::Storage(Storage<T> &&rhs) :
	m_ptr(rhs.m_ptr),
	m_size(rhs.m_size),
	m_capacity(rhs.m_capacity)
{
	rhs.m_ptr		= nullptr;
	rhs.m_size		= 0;
	rhs.m_capacity	= 0;
}

template <typename T>
Storage<T>::Storage(const std::initializer_list<T> &values) :
	m_ptr(new T[values.size()]),
	m_size(values.size()),
	m_capacity(values.size())
{
	size_t index = 0;
	for(const T &value : values)
	{
		m_ptr[index] = value;
		++index;
	}
}

template <typename T>
Storage<T>::Storage(const Serialized &node) :
	m_ptr(new T[node.size()]),
	m_size(node.size()),
	m_capacity(node.size())
{
	node.get(begin(), end());
}

template <typename T>
Storage<T>::~Storage()
{
	delete[] m_ptr;
}

template <typename T>
Storage<T> &Storage<T>::operator=(const Storage<T> &copy)
{
	if(this != &copy)
	{
		resize(copy.size());
		size_t index = 0;
		for(const T &value : copy)
		{
			m_ptr[index] = value;
			++index;
		}
	}
	return *this;
}

template <typename T>
Storage<T> &Storage<T>::operator=(const std::initializer_list<T> &values)
{
	resize(values.size());
	size_t index = 0;
	for(const T &value : values)
	{
		m_ptr[index] = value;
		++index;
	}
	return *this;
}

template <typename T>
Storage<T> &Storage<T>::resize(size_t n, const T &defaultValue)
{
	reserve(n);
	for(size_t i = m_size; i < n; ++i)
		m_ptr[i] = defaultValue;
	m_size = n;
	return *this;
}

template <typename T>
Storage<T> &Storage<T>::reserve(size_t n)
{
	if(n > m_capacity)
	{
		T *ptr = new T[n];
		for(size_t i = 0; i < m_size; ++i)
			ptr[i] = m_ptr[i];
		delete[] m_ptr;
		m_ptr = ptr;
		m_capacity = n;
	}
	return *this;
}

template <typename T>
Storage<T> &Storage<T>::push_back(const T &value)
{
	resize(m_size + 1);
	m_ptr[m_size - 1] = value;
	return *this;
}

template <typename T>
Storage<T> &Storage<T>::pop_back()
{
	--m_size;
	return *this;
}

template <typename T>
Storage<T> &Storage<T>::append(const Storage<T> &other)
{
	reserve(m_size + other.m_size);
	for(const T &value : other)
		push_back(value);
	return *this;
}

template <typename T>
Storage<T> &Storage<T>::erase(size_t index)
{
	NNAssertLessThan(index, m_size, "Attempted to erase an index that is out of bounds!");
	for(size_t i = index + 1; i < m_size; ++i)
	{
		m_ptr[i - 1] = m_ptr[i];
	}
	--m_size;
	return *this;
}

template <typename T>
Storage<T> &Storage<T>::clear()
{
	m_size = 0;
	return *this;
}

template <typename T>
T *Storage<T>::ptr()
{
	return m_ptr;
}

template <typename T>
const T *Storage<T>::ptr() const
{
	return m_ptr;
}

template <typename T>
size_t Storage<T>::size() const
{
	return m_size;
}

template <typename T>
bool Storage<T>::operator==(const Storage<T> &other) const
{
	if(this == &other)
	{
		return true;
	}
	if(m_size != other.m_size)
	{
		return false;
	}
	for(size_t i = 0; i < other.m_size; ++i)
	{
		if(m_ptr[i] != other.m_ptr[i])
		{
			return false;
		}
	}
	return true;
}

template <typename T>
bool Storage<T>::operator!=(const Storage<T> &other) const
{
	return !(*this == other);
}

template <typename T>
T &Storage<T>::at(size_t i)
{
	NNAssertLessThan(i, m_size, "Attempted to access an index that is out of bounds!");
	return m_ptr[i];
}

template <typename T>
const T &Storage<T>::at(size_t i) const
{
	NNAssertLessThan(i, m_size, "Attempted to access an index that is out of bounds!");
	return m_ptr[i];
}

template <typename T>
T &Storage<T>::operator[](size_t i)
{
	NNAssertLessThan(i, m_size, "Attempted to access an index that is out of bounds!");
	return m_ptr[i];
}

template <typename T>
const T &Storage<T>::operator[](size_t i) const
{
	NNAssertLessThan(i, m_size, "Attempted to access an index that is out of bounds!");
	return m_ptr[i];
}

template <typename T>
T &Storage<T>::front()
{
	NNAssertGreaterThan(m_size, 0, "Attempted to access an index that is out of bounds!");
	return *m_ptr;
}

template <typename T>
const T &Storage<T>::front() const
{
	NNAssertGreaterThan(m_size, 0, "Attempted to access an index that is out of bounds!");
	return *m_ptr;
}

template <typename T>
T &Storage<T>::back()
{
	NNAssertGreaterThan(m_size, 0, "Attempted to access an index that is out of bounds!");
	return m_ptr[m_size - 1];
}

template <typename T>
const T &Storage<T>::back() const
{
	NNAssertGreaterThan(m_size, 0, "Attempted to access an index that is out of bounds!");
	return m_ptr[m_size - 1];
}

template <typename T>
T *Storage<T>::begin()
{
	return m_ptr;
}

template <typename T>
const T *Storage<T>::begin() const
{
	return m_ptr;
}

template <typename T>
T *Storage<T>::end()
{
	return m_ptr + m_size;
}

template <typename T>
const T *Storage<T>::end() const
{
	return m_ptr + m_size;
}

template <typename T>
void Storage<T>::save(Serialized &node) const
{
	node.set(begin(), end());
}

}

#endif
