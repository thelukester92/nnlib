#ifndef VECTOR_H
#define VECTOR_H

#include <memory>

namespace nnlib
{

/// A 1-dimensional tensor.
/// The default copy behavior is a shallow copy unless using an explicit copy method.
template <typename T = double>
class Vector
{
public:
	Vector(size_t size = 0, const T &defaultValue = 0) :
		m_ptr(new T[size]),
		m_size(size),
		m_stride(1),
		m_shared(m_ptr)
	{
		fill(defaultValue);
	}
	
	Vector(Vector &v) :
		m_ptr(v.m_ptr),
		m_size(v.m_size),
		m_stride(v.m_stride),
		m_shared(v.m_shared)
	{}
	
	// MARK: Vector operations
	
	Vector &fill(const T &value)
	{
		for(size_t i = 0, end = m_size * m_stride; i < end; i += m_stride)
			m_ptr[i] = value;
		return *this;
	}
	
	Vector &resize(size_t size, const T &defaultValue = 0)
	{
		if(size > m_size)
		{
			T *ptr = new T[size];
			for(size_t i = 0; i < m_size; ++i)
				ptr[i] = m_ptr[i * m_stride];
			for(size_t i = m_size; i < size; ++i)
				ptr[i] = defaultValue;
			m_stride = 1;
			
			NNWarn(m_shared.use_count() == 1, "Vector::resize decoupled shared data!");
			m_shared.reset(m_ptr = ptr);
		}
		return *this;
	}
	
	// MARK: Element access
	
	T &operator[](size_t i)
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_ptr[i * m_stride];
	}
	
	T &operator[](size_t i) const
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_ptr[i * m_stride];
	}
	
	T &operator()(size_t i)
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_ptr[i * m_stride];
	}
	
	T &operator()(size_t i) const
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_ptr[i * m_stride];
	}
	
private:
	T *m_ptr;						///< Pointer to the underlying data, with offset included.
	size_t m_size;					///< Number of elements in this vector.
	size_t m_stride;				///< Distance between elements in this vector.
	std::shared_ptr<T> m_shared;	///< The shared data, for ARC.
};

}

#endif
