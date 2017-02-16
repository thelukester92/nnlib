#ifndef BATCHER_H
#define BATCHER_H

#include "random.h"
#include "matrix.h"

namespace nnlib
{

template <typename T = double>
class Batcher
{
public:
	Batcher(Matrix<T> &feat, Matrix<T> &lab, size_t batchSize = 1)
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

}

#endif
