#ifndef CORE_STORAGE_HPP
#define CORE_STORAGE_HPP

#include <initializer_list>
#include "../serialization/serialized.hpp"

namespace nnlib
{

/// Unique, contigious storage that manages its own memory.
/// May be shared across multiple objects.
/// Used by tensors.
template <typename T>
class Storage
{
public:
	Storage(size_t n = 0, const T &defaultValue = T()) :
		m_ptr(new T[n]),
		m_size(n),
		m_capacity(n)
	{
		for(size_t i = 0; i < n; ++i)
			m_ptr[i] = defaultValue;
	}
	
	Storage(const Storage &copy) :
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
	
	Storage(Storage &&rhs) :
		m_ptr(rhs.m_ptr),
		m_size(rhs.m_size),
		m_capacity(rhs.m_capacity)
	{
		rhs.m_ptr		= nullptr;
		rhs.m_size		= 0;
		rhs.m_capacity	= 0;
	}
	
	Storage(const std::initializer_list<T> &values) :
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
	
	/// Load from a serialized node.
	Storage(const Serialized &node) :
		m_ptr(new T[node.size()]),
		m_size(node.size()),
		m_capacity(node.size())
	{
		node.get(begin(), end());
	}
	
	~Storage()
	{
		delete[] m_ptr;
	}
	
	Storage &operator=(const Storage &copy)
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
	
	Storage &operator=(const std::initializer_list<T> &values)
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
	
	Storage &resize(size_t n, const T &defaultValue = T())
	{
		reserve(n);
		for(size_t i = m_size; i < n; ++i)
			m_ptr[i] = defaultValue;
		m_size = n;
		return *this;
	}
	
	Storage &reserve(size_t n)
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
	
	Storage &push_back(const T &value)
	{
		resize(m_size + 1);
		m_ptr[m_size - 1] = value;
		return *this;
	}
	
	Storage &pop_back()
	{
		--m_size;
		return *this;
	}
	
	Storage &append(const Storage &other)
	{
		reserve(m_size + other.m_size);
		for(const T &value : other)
			push_back(value);
		return *this;
	}
	
	Storage &erase(size_t index)
	{
		NNAssertLessThan(index, m_size, "Attempted to erase an index that is out of bounds!");
		for(size_t i = index + 1; i < m_size; ++i)
		{
			m_ptr[i - 1] = m_ptr[i];
		}
		--m_size;
		return *this;
	}
	
	Storage &clear()
	{
		m_size = 0;
		return *this;
	}
	
	T *ptr()
	{
		return m_ptr;
	}
	
	const T *ptr() const
	{
		return m_ptr;
	}
	
	size_t size() const
	{
		return m_size;
	}
	
	bool operator==(const Storage &other) const
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
	
	bool operator!=(const Storage &other) const
	{
		return !(*this == other);
	}
	
	// MARK: Element access
	
	T &operator[](size_t i)
	{
		NNAssertLessThan(i, m_size, "Attempted to access an index that is out of bounds!");
		return m_ptr[i];
	}
	
	const T &operator[](size_t i) const
	{
		NNAssertLessThan(i, m_size, "Attempted to access an index that is out of bounds!");
		return m_ptr[i];
	}
	
	T &front()
	{
		NNAssertGreaterThan(m_size, 0, "Attempted to access an index that is out of bounds!");
		return *m_ptr;
	}
	
	const T &front() const
	{
		NNAssertGreaterThan(m_size, 0, "Attempted to access an index that is out of bounds!");
		return *m_ptr;
	}
	
	T &back()
	{
		NNAssertGreaterThan(m_size, 0, "Attempted to access an index that is out of bounds!");
		return m_ptr[m_size - 1];
	}
	
	const T &back() const
	{
		NNAssertGreaterThan(m_size, 0, "Attempted to access an index that is out of bounds!");
		return m_ptr[m_size - 1];
	}
	
	// MARK: Iterators
	
	T *begin()
	{
		return m_ptr;
	}
	
	const T *begin() const
	{
		return m_ptr;
	}
	
	T *end()
	{
		return m_ptr + m_size;
	}
	
	const T *end() const
	{
		return m_ptr + m_size;
	}
	
	/// Save to a serialized node.
	void save(Serialized &node) const
	{
		node.set(begin(), end());
	}
	
private:
	T *m_ptr;			///< The data itself.
	size_t m_size;		///< Number of elements being used.
	size_t m_capacity;	///< Number of elements available in buffer.
};

}

#endif
