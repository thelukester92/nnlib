#ifndef UTIL_TENSOR_ITERATOR_HPP
#define UTIL_TENSOR_ITERATOR_HPP

#include "../core/tensor.hpp"

namespace nnlib
{

namespace detail
{
	
}

template <typename T>
class TensorIterator : public std::iterator<std::forward_iterator_tag, T, std::ptrdiff_t, const T *, T &>
{
using TT = typename std::remove_const<T>::type;
public:
	TensorIterator(const Tensor<TT> *tensor, bool end = false) :
		m_contiguous(tensor->contiguous()),
		m_shape(tensor->shape()),
		m_stride(tensor->strides()),
		m_indices(m_contiguous ? 1 : tensor->dims(), 0),
		m_ptr(const_cast<Tensor<TT> *>(tensor)->ptr())
	{
		if(end || tensor->size() == 0)
		{
			m_indices[0] = m_shape[0];
			m_ptr += m_stride[0] * m_indices[0];
		}
	}
	
	TensorIterator &operator++()
	{
		if(m_contiguous)
		{
			++m_ptr;
			return *this;
		}
		
		size_t d = m_indices.size() - 1;
		++m_indices[d];
		m_ptr += m_stride[d];
		
		while(m_indices[d] >= m_shape[d] && d > 0)
		{
			m_ptr -= m_stride[d] * m_indices[d];
			m_indices[d] = 0;
			
			--d;
			
			++m_indices[d];
			m_ptr += m_stride[d];
		}
		
		return *this;
	}
	
	TensorIterator operator++(int)
	{
		TensorIterator it = *this;
		++(*this);
		return it;
	}
	
	T &operator*()
	{
		return *m_ptr;
	}
	
	bool operator==(const TensorIterator &other)
	{
		return !(*this != other);
	}
	
	bool operator!=(const TensorIterator &other)
	{
		if(m_contiguous)
			return m_ptr != other.m_ptr;
		else
			return m_indices != other.m_indices;
	}
	
private:
	bool m_contiguous;
	const Storage<size_t> &m_shape;
	const Storage<size_t> &m_stride;
	Storage<size_t> m_indices;
	TT *m_ptr;
};

}

#endif
