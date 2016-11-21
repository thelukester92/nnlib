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
class RandomBatchIterator
{
public:
	/// General constructor. Starts in the past-the-end state.
	RandomBatchIterator(Matrix<T> &features, Matrix<T> &labels, size_t n = 1)
	: m_random(), m_features(features), m_labels(labels), m_featureBatch(m_features, n, features.cols()), m_labelBatch(m_labels, n, labels.cols()), m_batchSize(n), m_offset(m_features.rows())
	{
		NNAssert(m_features.rows() == m_labels.rows(), "Features and labels must have the same number of rows!");
	}
	
	/// Shuffle the matrix and reset offset to 0.
	void reset()
	{
		m_offset = 0;
		m_featureBatch.resize(m_batchSize, m_features.cols());
		m_labelBatch.resize(m_batchSize, m_labels.cols());
		size_t n = m_features.rows();
		for(size_t i = n - 1; i > 0; --i)
		{
			size_t j = m_random.uniformInt(i);
			m_features.swapRows(i, j);
			m_labels.swapRows(i, j);
		}
	}
	
	/// The current mini-batch of features.
	Matrix<T> &features()
	{
		NNAssert(m_offset < m_features.rows(), "Cannot batch an uninitialized batch iterator!");
		return m_featureBatch;
	}
	
	/// The current mini-batch of labels.
	Matrix<T> &labels()
	{
		NNAssert(m_offset < m_labels.rows(), "Cannot batch an uninitialized batch iterator!");
		return m_labelBatch;
	}
	
	/// The current mini-batch.
	std::pair<Matrix<T>, Matrix<T>> &operator*()
	{
		NNAssert(m_offset < m_features.rows(), "Cannot batch an uninitialized batch iterator!");
		return std::make_pair(m_featureBatch, m_labelBatch);
	}
	
	/// std-like iterator begin.
	RandomBatchIterator &begin()
	{
		reset();
		return *this;
	}
	
	/// std-like iterator end.
	RandomBatchIterator end()
	{
		return RandomBatchIterator(m_features, m_labels, m_batchSize);
	}
	
	RandomBatchIterator &operator++()
	{
		m_offset += m_batchSize;
		if(m_offset < m_features.rows())
		{
			m_featureBatch.setOffset(m_offset);
			m_labelBatch.setOffset(m_offset);
		}
		if(m_offset + m_batchSize > m_features.rows())
		{
			m_featureBatch.resize(m_features.rows() - m_offset, m_features.cols());
			m_labelBatch.resize(m_labels.rows() - m_offset, m_labels.cols());
		}
		return *this;
	}
	
	RandomBatchIterator operator++(int)
	{
		RandomBatchIterator it(*this);
		++*this;
		return it;
	}
	
	bool operator==(const RandomBatchIterator &ri)
	{
		return m_offset == ri.m_offset && m_batchSize == ri.m_batchSize && &m_features == &ri.m_features && &m_labels == &ri.m_labels;
	}
	
	bool operator!=(const RandomBatchIterator &ri)
	{
		return !(*this == ri);
	}
private:
	Random m_random;
	Matrix<T> &m_features, &m_labels;
	Matrix<T> m_featureBatch, m_labelBatch;
	size_t m_batchSize, m_offset;
};

}

#endif
