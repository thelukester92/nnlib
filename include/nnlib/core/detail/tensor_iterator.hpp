#ifndef CORE_TENSOR_ITERATOR_HPP
#define CORE_TENSOR_ITERATOR_HPP

#include "../tensor.hpp"

namespace nnlib
{

template <typename T>
class TensorIterator : public std::iterator<std::forward_iterator_tag, T, std::ptrdiff_t, const T *, T &>
{
using TT = typename std::remove_const<T>::type;
public:
	TensorIterator(const Tensor<TT> *tensor, bool end = false);
	TensorIterator &operator++();
	TensorIterator operator++(int);
	T &operator*();
	bool operator==(const TensorIterator &other);
	bool operator!=(const TensorIterator &other);

private:
	bool m_contiguous;
	const Storage<size_t> &m_shape;
	const Storage<size_t> &m_stride;
	Storage<size_t> m_indices;
	TT *m_ptr;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::TensorIterator<NN_REAL_T>;
	extern template class nnlib::TensorIterator<const NN_REAL_T>;
#elif !defined NN_IMPL
	#include "tensor_iterator.tpp"
#endif

#endif
