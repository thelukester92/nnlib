#ifndef TENSOR_H
#define TENSOR_H

namespace nnlib
{

#include "error.h"
#include "storage.h"

template <typename T>
class Tensor
{
public:
	template <typename ... Ts>
	Tensor(Ts... dims)
	{
		resize(dims...);
	}
	
	template <typename ... Ts>
	void resize(Ts... dims)
	{
		m_dims = { static_cast<size_t>(dims)... };
		size_t product = 1;
		for(size_t i = 0, j = m_dims.size(); i < j; ++i)
		{
			product *= m_dims[i];
		}
		m_data.resize(product);
	}
	
	size_t size() const
	{
		return m_data.size();
	}
	
	size_t size(size_t dim) const
	{
		return m_dims[dim];
	}
	
	template <typename ... Ts>
	T &operator()(Ts... indices)
	{
		return m_data[indexOf({ static_cast<size_t>(indices)... })];
	}
private:
	Storage<size_t> m_dims;
	Storage<T> m_data;
	
	/// a 5x3x2 tensor accessed at 4, 2, 1 should produce:
	/// 4 * (3*2) + 2 * (2) + 1
	/// = 1 + 2 * (3 * 4 + 2)
	/// = (((4 * 3) + 2) * 2) + 1
	/// = 29
	
	size_t indexOf(const std::initializer_list<size_t> &indices)
	{
		size_t result = 0, dim = 1;
		const size_t *i, *j;
		for(i = indices.begin(), j = indices.end() - 1; i != j; ++i, ++dim)
		{
			result += *i;
			result *= m_dims[dim];
		}
		return result + *i;
	}
};

}

#endif
