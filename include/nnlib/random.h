#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include "matrix.h"

namespace nnlib
{

template <typename T = double>
class Random
{
public:
	Random() = delete;
	
	static void seed(size_t seed = 0)
	{
		m_engine.seed(0);
	}
	
	static T uniform(T to)
	{
		return std::uniform_real_distribution<T>(0, to)(m_engine);
	}
	
	static T uniform(T from, T to)
	{
		return std::uniform_real_distribution<T>(from, to)(m_engine);
	}
	
	static T normal(T mean = 0.0, T stddev = 1.0)
	{
		return std::normal_distribution<T>(mean, stddev)(m_engine);
	}
	
	static T normal(T mean, T stddev, T cap)
	{
		double n;
		std::normal_distribution<T> dist(mean, stddev);
		do
		{
			n = dist(m_engine);
		}
		while(fabs(n - mean) > cap);
		return n;
	}
private:
	static std::default_random_engine m_engine;
};

template <typename T>
std::default_random_engine Random<T>::m_engine = std::default_random_engine(std::random_device()());

template <>
class Random<size_t>
{
using T = size_t;
public:
	Random() = delete;
	
	static void seed(size_t seed = 0)
	{
		m_engine.seed(0);
	}
	
	static T uniform(T to)
	{
		return std::uniform_int_distribution<T>(0, to)(m_engine);
	}
	
	static T uniform(T from, T to)
	{
		return std::uniform_int_distribution<T>(from, to)(m_engine);
	}
private:
	static std::default_random_engine m_engine;
};

std::default_random_engine Random<size_t>::m_engine = std::default_random_engine(std::random_device()());

template <typename T = double>
class Batcher
{
public:
	Batcher(Matrix<T> &feat, Matrix<T> &lab, size_t batchSize)
	: m_feat(feat), m_lab(lab),
	  m_batchFeat(feat.block(0, 0, batchSize)), m_batchLab(lab.block(0, 0, batchSize)),
	  m_batches((size_t) floor(feat.rows() / double(batchSize))), m_index(0), m_batchSize(batchSize)
	{
		reset();
	}
	
	Batcher &reset()
	{
		Matrix<T>::shuffleRows(m_feat, m_lab);
		m_index = 0;
		return *this;
	}
	
	bool next(bool wrap = false)
	{
		if(m_index >= m_batches)
		{
			if(wrap)
				reset();
			else
				return false;
		}
		
		m_feat.block(m_batchFeat, m_batchSize * m_index);
		m_lab.block(m_batchLab, m_batchSize * m_index);
		++m_index;
		
		return true;
	}
	
	Matrix<T> &features()
	{
		return m_batchFeat;
	}
	
	Matrix<T> &labels()
	{
		return m_batchLab;
	}
private:
	Matrix<T> &m_feat, &m_lab;
	Matrix<T> m_batchFeat, m_batchLab;
	size_t m_batches, m_index, m_batchSize;
};

template <typename T = double>
class BlockBatcher
{
public:
	BlockBatcher(Matrix<T> &feat, Matrix<T> &lab, size_t batchSize, size_t blockSize)
	: m_feat(feat), m_lab(lab),
	  m_batchFeat(feat.block(0, 0, batchSize * blockSize)), m_batchLab(lab.block(0, 0, batchSize)),
	  m_batches((size_t) floor(feat.rows() / double(batchSize))), m_index(0), m_batchSize(batchSize), m_blockSize(blockSize)
	{
		reset();
	}
	
	BlockBatcher &reset()
	{
		Matrix<T>::shuffleRows(m_feat, m_lab, m_blockSize);
		m_index = 0;
		return *this;
	}
	
	bool next(bool wrap = false)
	{
		if(m_index >= m_batches)
		{
			if(wrap)
				reset();
			else
				return false;
		}
		
		m_feat.block(m_batchFeat, m_batchSize * m_blockSize * m_index);
		m_lab.block(m_batchLab, m_batchSize * m_index);
		++m_index;
		
		return true;
	}
	
	Matrix<T> &features()
	{
		return m_batchFeat;
	}
	
	Matrix<T> &labels()
	{
		return m_batchLab;
	}
private:
	Matrix<T> &m_feat, &m_lab;
	Matrix<T> m_batchFeat, m_batchLab;
	size_t m_batches, m_index, m_batchSize, m_blockSize;
};

}

#endif
