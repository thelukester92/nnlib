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
template <typename T = double>
class Vector : public Tensor<T>
{
typedef typename std::remove_const<T>::type TT;
friend class Vector<const T>;
friend class Vector<TT>;
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
		NNAssert(A.m_size == B.m_size, "Incompatible vectors for copying!");
		Algebra<T>::instance().copy(A.m_size, A.m_ptr, A.m_stride, B.m_ptr, B.m_stride);
	}
	
	/// Vector-scalar multiplication.
	static void multiply(T scalar, Vector &A)
	{
		Algebra<T>::instance().scal(A.m_size, scalar, A.m_ptr, A.m_stride);
	}
	
	/// Element-wise / Pointwise / Hadamard product.
	static void pointwiseProduct(const Vector &u, const Vector &v, Vector &p)
	{
		NNAssert(u.size() == v.size() && v.size() == p.size(), "Incompatible vectors for pointwise product!");
		auto i = u.begin(), j = v.begin();
		for(T &val : p)
		{
			val = *i++ * *j++;
		}
	}
	
	/// Dot product of two vectors.
	static T dotProduct(const Vector &u, const Vector &v)
	{
		NNAssert(u.size() == v.size(), "Incompatible vectors for dot product!");
		T sum = 0;
		auto i = u.begin(), j = v.begin(), k = v.end();
		while(j != k)
		{
			sum += *i++ * *j++;
		}
		return sum;
	}
	
	// MARK: Factory methods
	
	/// Concatenate (deep copy) several tensors into this vector.
	/// \todo variadic template args
	/*
	static Vector concatenate(Vector<Tensor<T> *> tensors)
	{
		size_t size = 0;
		for(Tensor<T> *t : tensors)
			size += t->size();
		Vector v(size);
		
		auto i = v.begin();
		for(Tensor<T> *t : tensors)
			for(T &v : *t)
				*i++ = v;
		
		return v;
	}
	*/
	
	/// Create a vector (flattened) from several tensors.
	/// \todo realize that this assumes contiguous data; we should enforce this with an assert
	static Vector flatten(Vector<Tensor<T> *> tensors)
	{
		size_t size = 0;
		for(Tensor<T> *t : tensors)
			size += t->size();
		Vector v(size);
		
		T *ptr = v.m_ptr;
		for(Tensor<T> *t : tensors)
		{
			T *tPtr = t->ptr();
			for(size_t i = 0; i < t->size(); ++i)
				ptr[i] = tPtr[i];
			t->set(ptr, t->size(), v.m_shared);
			ptr += t->size();
		}
		
		return v;
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
	
	// MARK: Non-static Algebra
	
	/// Deep copy the contents of another vector.
	Vector &copy(const Vector &A)
	{
		Vector::copy(A, *this);
		return *this;
	}
	
	/// Add another vector, scaled.
	Vector &addScaled(const Vector &A, T scalar = 1.0)
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
	
	/// Element-wise / Pointwise / Hadamard product, storing the result in this.
	Vector &pointwiseProduct(const Vector &v)
	{
		NNAssert(v.m_size == m_size, "Incompatible vector for pointwise product!");
		auto itr = v.begin();
		for(T &val : *this)
		{
			val *= *itr++;
		}
		return *this;
	}
	
	/// Concatenate (deep copy) several tensors into this vector.
	/// \todo variadic template args
	/// \todo realize that this assumes contiguous data; we should enforce this with an assert
	Vector &concatenate(Vector<Tensor<T> *> tensors)
	{
		size_t size = 0;
		for(Tensor<T> *t : tensors)
			size += t->size();
		resize(size);
		
		auto i = begin();
		for(Tensor<T> *t : tensors)
		{
			T *tPtr = t->ptr();
			for(size_t j = 0; j < t->size(); ++i, ++j)
				*i = tPtr[j];
		}
		
		return *this;
	}
	
	/// Get a new vector that shares the same storage.
	Vector narrow(size_t size, size_t offset = 0)
	{
		return Vector(*this, offset * m_stride, size, m_stride);
	}
	
	/// Dot product with another vector.
	TT dotProduct(const Vector &v) const
	{
		NNAssert(v.size() == m_size, "Incompatible vector for dot product!");
		TT sum = 0;
		auto itr = v.begin();
		for(TT val : *this)
			sum += val * *itr++;
		return sum;
	}
	
	/// Squared distance to another vector.
	TT squaredDistance(const Vector &v)
	{
		NNAssert(v.size() == m_size, "Incompatible vector for distance!");
		TT distance = 0;
		auto itr = v.begin();
		for(TT val : *this)
		{
			distance += (val - *itr) * (val - *itr);
			++itr;
		}
		return distance;
	}
	
	// MARK: Statistics
	
	/// Add all the elements of this vector and return the sum.
	TT sum() const
	{
		TT d = 0;
		for(TT val : *this)
			d += val;
		return d;
	}
	
	/// Get the minimum value.
	TT minimum() const
	{
		TT smallest = *begin();
		for(TT val : *this)
			if(val < smallest)
				smallest = val;
		return smallest;
	}
	
	/// Get the maximum value.
	TT maximum() const
	{
		TT biggest = *begin();
		for(TT val : *this)
			if(val > biggest)
				biggest = val;
		return biggest;
	}
	
	/// Get the average value.
	TT mean() const
	{
		return sum() / m_size;
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
	
	// MARK: Iterators
	
	Iterator begin()
	{
		return Iterator(m_ptr, m_stride);
	}
	
	Iterator begin() const
	{
		return Iterator(m_ptr, m_stride);
	}
	
	Iterator end()
	{
		return Iterator(m_ptr + m_stride * m_size, m_stride);
	}
	
	Iterator end() const
	{
		return Iterator(m_ptr + m_stride * m_size, m_stride);
	}
private:
	size_t m_stride;	///< The stride between elements in this Vector
};

}

#endif
