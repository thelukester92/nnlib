#ifndef BATCHER_H
#define BATCHER_H

#include "random.h"
#include "tensor.h"

namespace nnlib
{

/// Takes two tensors and returns random slices along the major dimensions, one slice at at time.
/// This is useful for optimization.
/// Batcher requires non-const inputs and will shuffle them unless the copy flag is true.
template <typename T = double>
class Batcher
{
public:
	Batcher(Tensor<T> &feat, Tensor<T> &lab, size_t bats = 1, bool copy = false) :
		m_feat(copy ? feat.copy() : feat),
		m_lab(copy ? lab.copy() : lab),
		m_featBatch(m_feat),
		m_labBatch(m_lab),
		m_batch(bats)
	{
		NNAssert(feat.size(0) == lab.size(0), "Incompatible features and labels!");
		NNAssert(bats <= feat.size(0), "Invalid batch size!");
		reset();
	}
	
	Batcher &batch(size_t bats)
	{
		NNAssert(bats <= m_feat.size(0), "Invalid batch size!");
		m_batch = bats;
		reset();
		return *this;
	}
	
	size_t batch() const
	{
		return m_batch;
	}
	
	size_t batches() const
	{
		return m_feat.size(0) / m_batch;
	}
	
	Batcher &reset()
	{
		m_offset = 0;
		for(size_t i = 0, end = m_feat.size(0); i < end; ++i)
		{
			size_t j = Random<size_t>::uniform(end);
			m_feat.select(0, i).swap(m_feat.select(0, j));
			m_lab.select(0, i).swap(m_lab.select(0, j));
		}
		
		m_feat.sub(m_featBatch, { { m_offset, m_batch }, {} });
		m_lab.sub(m_labBatch, { { m_offset, m_batch }, {} });
		
		return *this;
	}
	
	bool next(bool autoReset = false)
	{
		m_offset += m_batch;
		if(m_offset + m_batch > m_feat.size(0))
		{
			if(autoReset)
			{
				reset();
			}
			else
			{
				return false;
			}
		}
		
		m_feat.sub(m_featBatch, { { m_offset, m_batch }, {} });
		m_lab.sub(m_labBatch, { { m_offset, m_batch }, {} });
		
		return true;
	}
	
	Tensor<T> &features()
	{
		return m_featBatch;
	}
	
	Tensor<T> &labels()
	{
		return m_labBatch;
	}
	
private:
	Tensor<T> m_feat;
	Tensor<T> m_lab;
	Tensor<T> m_featBatch;
	Tensor<T> m_labBatch;
	size_t m_offset;
	size_t m_batch;
};

/// This variation of Batcher yields sequences of batches (for sequential data).
/// Unlike the regular Batcher, the SequenceBatcher only yields one batch per reset,
/// so there is no "next" method.
template <typename T = double>
class SequenceBatcher
{
public:
	SequenceBatcher(const Tensor<T> &feat, const Tensor<T> &lab, size_t seqLen = 1, size_t bats = 1) :
		m_feat(feat),
		m_lab(lab),
		m_featBatch(seqLen, bats, m_feat.size(1)),
		m_labBatch(seqLen, bats, m_lab.size(1)),
		m_batch(bats),
		m_seqLen(seqLen)
	{
		NNAssert(feat.dims() == 2 && lab.dims() == 2, "SequenceBatcher only works with matrix inputs!");
		NNAssert(feat.size(0) == lab.size(0), "Incompatible features and labels!");
		NNAssert(bats <= feat.size(0), "Invalid batch size!");
		reset();
	}
	
	SequenceBatcher &seqLen(size_t seqLen)
	{
		m_seqLen = seqLen;
		reset();
		return *this;
	}
	
	size_t seqLen() const
	{
		return m_seqLen;
	}
	
	SequenceBatcher &batch(size_t bats)
	{
		NNAssert(bats <= m_feat.size(0), "Invalid batch size!");
		m_batch = bats;
		reset();
		return *this;
	}
	
	size_t batch() const
	{
		return m_batch;
	}
	
	SequenceBatcher &reset()
	{
		Storage<size_t> indices(m_batch);
		for(size_t &index : indices)
		{
			index = Random<size_t>::uniform(m_feat.size(0) - m_seqLen + 1);
		}
		
		for(size_t i = 0; i < m_seqLen; ++i)
		{
			for(size_t j = 0; j < m_batch; ++j)
			{
				m_featBatch.sub({ { i }, { j }, {} }).copy(m_feat.sub({ { indices[j] }, {} }));
				m_labBatch.sub({ { i }, { j }, {} }).copy(m_lab.sub({ { indices[j] }, {} }));
				++indices[j];
			}
		}
		
		return *this;
	}
	
	Tensor<T> &features()
	{
		return m_featBatch;
	}
	
	Tensor<T> &labels()
	{
		return m_labBatch;
	}
	
private:
	const Tensor<T> &m_feat;
	const Tensor<T> &m_lab;
	Tensor<T> m_featBatch;
	Tensor<T> m_labBatch;
	size_t m_batch;
	size_t m_seqLen;
};

}

#endif
