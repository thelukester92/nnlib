#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include "tensor.h"

namespace nnlib
{

class Random
{
public:
	Random();
	Random(size_t seed);
	
	double uniform(double a = 0.0, double b = 1.0);
	double normal(double mean = 0.0, double stddev = 1.0);
	double normal(double mean, double stddev, double cap);
	size_t uniformInt(size_t n = std::numeric_limits<size_t>::max());
	
	/// Set all elements in a tensor to random values drawn from a normal distribution.
	template <typename T>
	void fillNormal(Tensor<T> &t, double mean = 0.0, double stddev = 1.0, double cap = 3.0)
	{
		for(auto &i : t)
			i = normal(mean, stddev, cap);
	}
	
private:
	std::default_random_engine m_engine;
};

/// Randomly iterate through integers between 0 and n (exclusive).
class RandomIterator
{
public:
	RandomIterator(size_t n);
	void reset();
	size_t *begin();
	size_t *end();
private:
	Random m_random;
	Vector<size_t> m_buffer;
};

/// Randomly iterate through a Matrix, batching together a submatrix.
/// The input matrix will be shuffled.
template <typename T>
class RandomBatcher
{
public:
	struct Iterator
	{
		Iterator(RandomBatcher &_ri, size_t _offset = 0)
		: ri(_ri), features(ri.m_features, ri.m_batchSize, ri.m_features.cols()), labels(ri.m_labels, ri.m_batchSize, ri.m_labels.cols()), offset(_offset)
		{}
		
		Iterator(const Iterator &i)
		: ri(i.ri), features(ri.m_features, ri.m_batchSize, ri.m_features.cols()), labels(ri.m_labels, ri.m_batchSize, ri.m_labels.cols()), offset(i.offset)
		{}
		
		Iterator &operator++()
		{
			offset += ri.m_batchSize;
			if(offset < ri.m_features.rows())
			{
				features.setOffset(offset);
				labels.setOffset(offset);
				if(offset + ri.m_batchSize > ri.m_features.rows())
				{
					features.resize(features.rows() - offset, features.cols());
					labels.resize(labels.rows() - offset, labels.cols());
				}
			}
			else
				offset = ri.m_features.rows();
			return *this;
		}
		
		Iterator operator++(int)
		{
			Iterator it(*this);
			++*this;
			return it;
		}
		
		Iterator &operator*()
		{
			return *this;
		}
		
		bool operator==(const Iterator &i)
		{
			return &ri == &i.ri && offset == i.offset;
		}
		
		bool operator!=(const Iterator &i)
		{
			return &ri != &i.ri || offset != i.offset;
		}
		
		RandomBatcher &ri;
		Matrix<T> features, labels;
	
	private:
		size_t offset;
	};
	
	/// General constructor. Starts in the past-the-end state.
	RandomBatcher(Matrix<T> &features, Matrix<T> &labels, size_t n = 1)
	: m_random(), m_features(features), m_labels(labels), m_batchSize(n)
	{
		NNAssert(m_features.rows() == m_labels.rows(), "Features and labels must have the same number of rows!");
	}
	
	/// Shuffle the matrix.
	void reset()
	{
		size_t n = m_features.rows();
		for(size_t i = n - 1; i > 0; --i)
		{
			size_t j = m_random.uniformInt(i);
			m_features.swapRows(i, j);
			m_labels.swapRows(i, j);
		}
	}
	
	/// std-like iterator begin.
	Iterator begin()
	{
		reset();
		return Iterator(*this);
	}
	
	/// std-like iterator end.
	Iterator end()
	{
		return Iterator(*this, m_features.rows());
	}
private:
	Random m_random;
	Matrix<T> &m_features, &m_labels;
	Matrix<T> m_featureBatch, m_labelBatch;
	size_t m_batchSize, m_offset;
};

}

#endif
