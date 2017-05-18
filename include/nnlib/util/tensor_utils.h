#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "../tensor.h"

namespace nnlib
{

template <typename T>
class TensorIterator : public std::iterator<std::forward_iterator_tag, T, T, const T *, T &>
{
using TT = typename std::remove_const<T>::type;
public:
	TensorIterator(const Tensor<TT> *tensor, bool end = false) :
		m_tensor(const_cast<Tensor<TT> *>(tensor)),
		m_indices(tensor->dims(), 0)
	{
		/// \todo Make the right side of the `||` faster; `size()`` isn't cached, so this will hurt performance.
		if(end || m_tensor->size() == 0)
		{
			m_indices[0] = m_tensor->size(0);
		}
	}
	
	TensorIterator &advance(size_t dim = (size_t) -1)
	{
		dim = std::min(dim, m_tensor->dims());
		--m_indices.back();
		++m_indices[dim];
		return ++*this;
	}
	
	TensorIterator &operator++()
	{
		size_t dim = m_indices.size() - 1;
		++m_indices[dim];
		while(m_indices[dim] >= m_tensor->size(dim) && dim > 0)
		{
			m_indices[dim] = 0;
			--dim;
			++m_indices[dim];
		}
		return *this;
	}
	
	TensorIterator operator++(int)
	{
		TensorIterator it = *this;
		return ++it;
	}
	
	T &operator*()
	{
		return (*m_tensor)(m_indices);
	}
	
	bool operator==(const TensorIterator &other)
	{
		if(m_tensor != other.m_tensor)
		{
			return false;
		}
		
		for(size_t i = 0, j = m_indices.size(); i < j; ++i)
		{
			if(m_indices[i] != other.m_indices[i])
			{
				return false;
			}
		}
		
		return true;
	}
	
	bool operator !=(const TensorIterator &other)
	{
		return !(*this == other);
	}
private:
	Tensor<TT> *m_tensor;
	Storage<size_t> m_indices;
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const Tensor<T> &t)
{
	out << std::left << std::setprecision(5) << std::fixed;
	
	if(t.dims() == 1)
	{
		for(size_t i = 0; i < t.size(0); ++i)
		{
			out << t(i) << "\n";
		}
	}
	else if(t.dims() == 2)
	{
		for(size_t i = 0; i < t.size(0); ++i)
		{
			for(size_t j = 0; j < t.size(1); ++j)
			{
				out << std::setw(10) << t(i, j);
			}
			out << "\n";
		}
	}
	
	out << "[ Tensor of dimension " << t.size(0);
	for(size_t i = 1; i < t.dims(); ++i)
	{
		out << " x " << t.size(i);
	}
	out << " ]";
	
	return out;
}

}

#endif
