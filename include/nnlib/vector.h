#ifndef VECTOR_H
#define VECTOR_H

#include <type_traits>
#include <iterator>
#include <algorithm>
#include <initializer_list>
#include <utility>
#include "tensor.h"
#include "algebra.h"
#include "random.h"

namespace nnlib
{

template <typename T>
class Matrix;

/// Standard 1-dimensional tensor.
template <typename T>
class Vector : public Tensor<T>
{
friend class Vector<const T>;
friend class Vector<typename std::remove_const<T>::type>;
friend class Matrix<T>;
using Tensor<T>::m_ptr;
using Tensor<T>::m_size;
using Tensor<T>::m_shared;
public:
	// MARK: Iterator
	
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
	
	// MARK: Algebra; static methods
	
	/// Deep copy the contents of another vector.
	static void copy(const Vector &A, Vector &B)
	{
		Algebra<T>::instance().copy(A.m_size, A.m_ptr, A.m_stride, B.m_ptr, B.m_stride);
	}
	
	/// Vector-scalar multiplication.
	static void multiply(T scalar, Vector &A)
	{
		Algebra<T>::instance().scal(A.m_size, scalar, A.m_ptr, A.m_stride);
	}
	
	// MARK: Constructors
	
	/// Create a vector of size n.
	explicit Vector(size_t n = 0, const T &val = T()) : Tensor<T>(n), m_stride(1)
	{
		fill(val);
	}
	
	/// Create a shallow copy of another vector.
	Vector(const Vector &v) : Tensor<T>(v), m_stride(v.m_stride) {}
	
	/// Create a shallow copy of another tensor (i.e. matrix).
	Vector(const Tensor<T> &t, size_t offset, size_t size, size_t stride = 1) : Tensor<T>(t, offset, size), m_stride(stride) {}
	
	/// Create a const view of a non-const vector.
	template <typename U = T>
	Vector(const Vector<typename std::enable_if<std::is_const<U>::value, typename std::remove_const<U>::type>::type> &v) : Tensor<T>(v), m_stride(v.m_stride) {}
	
	/// Create a vector from an initializer list.
	Vector(const std::initializer_list<T> &l) : Tensor<T>(l.size()), m_stride(1)
	{
		size_t i = 0;
		for(const T &val : l)
			m_ptr[i++] = val;
	}
	
	/// Create a vector (flattened) from several tensors.
	Vector(Vector<Tensor<T> *> tensors) : Tensor<T>(0), m_stride(1)
	{
		size_t size = 0;
		for(Tensor<T> *t : tensors)
			size += t->size();
		resize(size);
		
		T *ptr = m_ptr;
		for(Tensor<T> *t : tensors)
		{
			T *tPtr = t->ptr();
			for(size_t i = 0; i < t->size(); ++i)
				ptr[i] = tPtr[i];
			t->set(ptr, t->size(), m_shared);
			ptr += t->size();
		}
	}
	
	// MARK: Non-static Algebra
	
	/// Add another vector, scaled.
	Vector &addScaled(const Vector &A, T scalar)
	{
		Algebra<T>::instance().axpy(m_size, scalar, A.m_ptr, A.m_stride, m_ptr, m_stride);
		return *this;
	}
	
	/// Multiply each element by a scalar.
	Vector &scale(T scalar)
	{
		Algebra<T>::instance().scal(m_size, scalar, m_ptr, m_stride);
		return *this;
	}
	
	/// Normalize between the given min and max.
	Vector &normalize(T min = 0.0, T max = 1.0)
	{
		T smallest = minimum(), biggest = maximum();
		for(T &val : *this)
			val = (val - smallest) / (biggest - smallest) * (max - min) + min;
		return *this;
	}
	
	/// Shuffle the elements of this vector.
	Vector &shuffle()
	{
		for(size_t i = m_size - 1; i > 0; --i)
		{
			size_t j = Random<size_t>::uniform(i);
			std::swap((*this)[i], (*this)[j]);
		}
		return *this;
	}
	
	// MARK: Element Manipulation
	
	Vector &fill(const T &val)
	{
		std::fill(begin(), end(), val);
		return *this;
	}
	
	Vector &resize(size_t size)
	{
		Tensor<T>::resize(size);
		return *this;
	}
	
	void push_back(const T &val)
	{
		resize(m_size + 1);
		m_ptr[(m_size - 1) * m_stride] = val;
	}
	
	size_t stride() const
	{
		return m_stride;
	}
	
	void erase(size_t i)
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		for(++i; i < m_size; ++i)
			m_ptr[(i - 1) * m_stride] = m_ptr[i * m_stride];
		--m_size;
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
	
	T &back()
	{
		return m_ptr[(m_size - 1) * m_stride];
	}
	
	const T &back() const
	{
		return m_ptr[(m_size - 1) * m_stride];
	}
	
	/// Get the minimum value.
	T minimum()
	{
		T smallest = *begin();
		for(T val : *this)
			if(val < smallest)
				smallest = val;
		return smallest;
	}
	
	/// Get the maximum value.
	T maximum()
	{
		T biggest = *begin();
		for(T val : *this)
			if(val > biggest)
				biggest = val;
		return biggest;
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
