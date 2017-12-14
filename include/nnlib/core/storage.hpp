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
	Storage(size_t n = 0, const T &defaultValue = T());
	Storage(const Storage &copy);
	Storage(Storage &&rhs);
	Storage(const std::initializer_list<T> &values);
	Storage(const Serialized &node);
	~Storage();
	
	Storage &operator=(const Storage &copy);
	Storage &operator=(const std::initializer_list<T> &values);
	
	Storage &resize(size_t n, const T &defaultValue = T());
	Storage &reserve(size_t n);
	
	Storage &push_back(const T &value);
	Storage &pop_back();
	Storage &append(const Storage &other);
	Storage &erase(size_t index);
	Storage &clear();
	
	T *ptr();
	const T *ptr() const;
	
	size_t size() const;
	
	bool operator==(const Storage &other) const;
	bool operator!=(const Storage &other) const;
	
	T &at(size_t i);
	const T &at(size_t i) const;
	T &operator[](size_t i);
	const T &operator[](size_t i) const;
	
	T &front();
	const T &front() const;
	T &back();
	const T &back() const;
	
	T *begin();
	const T *begin() const;
	T *end();
	const T *end() const;
	
	void save(Serialized &node) const;
	
private:
	T *m_ptr;			///< The data itself.
	size_t m_size;		///< Number of elements being used.
	size_t m_capacity;	///< Number of elements available in buffer.
};

}

#include "detail/storage.tpp"

#endif
