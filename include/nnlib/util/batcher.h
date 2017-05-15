#ifndef BATCHER_H
#define BATCHER_H

#include "random.h"
#include "tensor.h"

namespace nnlib
{

/// Takes two tensors and returns random slices along the major dimensions, one slice at at time.
/// This is useful for optimization.
/// Batcher requires non-const inputs and will shuffle them unless the copy flag is true.
class Batcher
{
public:
	Batcher(Tensor &feat, Tensor &lab, size_t bats = 1, bool copy = false) :
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
	
	Tensor &features()
	{
		return m_featBatch;
	}
	
	Tensor &labels()
	{
		return m_labBatch;
	}
	
private:
	Tensor m_feat;
	Tensor m_lab;
	Tensor m_featBatch;
	Tensor m_labBatch;
	size_t m_offset;
	size_t m_batch;
};

/// This variation of Batcher yields sequences of batches (for sequential data).
/// Unlike the regular Batcher, the SequenceBatcher only yields one sequence per reset,
/// so there is no "next" method.
template <typename T = double>
class SequenceBatcher
{
public:
	SequenceBatcher(const Tensor &feat, const Tensor &lab, size_t sequenceLength = 0, size_t bats = 1) :
		m_feat(feat),
		m_lab(lab),
		m_featBatch(sequenceLength, bats, m_feat.size(1)),
		m_labBatch(sequenceLength, bats, m_lab.size(1)),
		m_batch(bats),
		m_sequenceLength(sequenceLength)
	{
		NNAssert(feat.dims() == 2 && lab.dims() == 2, "SequenceBatcher only works with matrix inputs!");
		NNAssert(feat.size(0) == lab.size(0), "Incompatible features and labels (" + std::to_string(feat.size(0)) + " != " + std::to_string(lab.size(0)) + ")!");
		NNAssert(sequenceLength <= feat.size(0), "Invalid sequence length (" + std::to_string(sequenceLength) + " > " + std::to_string(feat.size(0)) + ")!");
		NNAssert(bats <= feat.size(0), "Invalid batch size (" + std::to_string(bats) + " > " + std::to_string(feat.size(0)) + ")!");
		reset();
	}
	
	SequenceBatcher &sequenceLength(size_t sequenceLength)
	{
		NNAssert(sequenceLength <= m_feat.size(0), "Invalid sequence length (" + std::to_string(sequenceLength) + " > " + std::to_string(m_feat.size(0)) + ")!");
		m_sequenceLength = sequenceLength;
		reset();
		return *this;
	}
	
	size_t sequenceLength() const
	{
		return m_sequenceLength;
	}
	
	SequenceBatcher &batch(size_t bats)
	{
		NNAssert(bats <= m_feat.size(0), "Invalid batch size (" + std::to_string(bats) + " > " + std::to_string(m_feat.size(0)) + ")!");
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
			index = Random<size_t>::uniform(m_feat.size(0) - m_sequenceLength + 1);
		}
		
		for(size_t i = 0; i < m_sequenceLength; ++i)
		{
			for(size_t j = 0; j < m_batch; ++j)
			{
				m_featBatch.sub({ { i }, { j }, {} }).copy(m_feat.narrow(0, indices[j]));
				m_labBatch.sub({ { i }, { j }, {} }).copy(m_lab.narrow(0, indices[j]));
				++indices[j];
			}
		}
		
		return *this;
	}
	
	Tensor &features()
	{
		return m_featBatch;
	}
	
	Tensor &labels()
	{
		return m_labBatch;
	}
	
private:
	const Tensor &m_feat;
	const Tensor &m_lab;
	Tensor m_featBatch;
	Tensor m_labBatch;
	size_t m_batch;
	size_t m_sequenceLength;
};

}

#endif
