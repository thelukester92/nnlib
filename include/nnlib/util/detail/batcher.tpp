#ifndef UTIL_BATCHER_TPP
#define UTIL_BATCHER_TPP

#include "../batcher.hpp"
#include "nnlib/math/random.hpp"

namespace nnlib
{

template <typename T>
Batcher<T>::Batcher(Tensor<T> &feat, Tensor<T> &lab, size_t bats, bool copy) :
    m_feat(copy ? feat.copy() : feat),
    m_lab(copy ? lab.copy() : lab),
    m_featBatch(m_feat),
    m_labBatch(m_lab),
    m_batch(bats)
{
    NNHardAssertEquals(feat.size(0), lab.size(0), "Incompatible features and labels!");
    NNHardAssertLessThanOrEquals(bats, feat.size(0), "Invalid batch size!");
    reset();
}

template <typename T>
Batcher<T>::Batcher(Tensor<T> &&feat, Tensor<T> &&lab, size_t bats, bool copy) :
    m_feat(copy ? feat.copy() : feat),
    m_lab(copy ? lab.copy() : lab),
    m_featBatch(m_feat),
    m_labBatch(m_lab),
    m_batch(bats)
{
    NNHardAssertEquals(feat.size(0), lab.size(0), "Incompatible features and labels!");
    NNHardAssertLessThanOrEquals(bats, feat.size(0), "Invalid batch size!");
    reset();
}

template <typename T>
Batcher<T>::Batcher(const Tensor<T> &feat, const Tensor<T> &lab, size_t bats) :
    m_feat(feat.copy()),
    m_lab(lab.copy()),
    m_featBatch(m_feat),
    m_labBatch(m_lab),
    m_batch(bats)
{
    NNHardAssertEquals(feat.size(0), lab.size(0), "Incompatible features and labels!");
    NNHardAssertLessThanOrEquals(bats, feat.size(0), "Invalid batch size!");
    reset();
}

template <typename T>
Batcher<T> &Batcher<T>::batch(size_t bats)
{
    NNAssertLessThanOrEquals(bats, m_feat.size(0), "Invalid batch size!");
    m_batch = bats;
    reset();
    return *this;
}

template <typename T>
size_t Batcher<T>::batch() const
{
    return m_batch;
}

template <typename T>
size_t Batcher<T>::batches() const
{
    return m_feat.size(0) / m_batch;
}

template <typename T>
Batcher<T> &Batcher<T>::reset()
{
    m_offset = 0;
    for(size_t i = 0, end = m_feat.size(0); i < end; ++i)
    {
        size_t j = Random<size_t>::sharedRandom().uniform(end);
        m_feat.select(0, i).swap(m_feat.select(0, j));
        m_lab.select(0, i).swap(m_lab.select(0, j));
    }

    m_feat.sub(m_featBatch, { { m_offset, m_batch }, {} });
    m_lab.sub(m_labBatch, { { m_offset, m_batch }, {} });

    return *this;
}

template <typename T>
bool Batcher<T>::next(bool autoReset)
{
    m_offset += m_batch;
    if(m_offset + m_batch > m_feat.size(0))
    {
        if(autoReset)
            reset();
        else
            return false;
    }

    m_feat.sub(m_featBatch, { { m_offset, m_batch }, {} });
    m_lab.sub(m_labBatch, { { m_offset, m_batch }, {} });

    return true;
}

template <typename T>
Tensor<T> &Batcher<T>::features()
{
    return m_featBatch;
}

template <typename T>
Tensor<T> &Batcher<T>::labels()
{
    return m_labBatch;
}

template <typename T>
Tensor<T> &Batcher<T>::allFeatures()
{
    return m_feat;
}

template <typename T>
Tensor<T> &Batcher<T>::allLabels()
{
    return m_lab;
}

template <typename T>
SequenceBatcher<T>::SequenceBatcher(const Tensor<T> &feat, const Tensor<T> &lab, size_t sequenceLength, size_t bats) :
    m_feat(feat),
    m_lab(lab),
    m_featBatch(sequenceLength, bats, m_feat.size(1)),
    m_labBatch(sequenceLength, bats, m_lab.size(1)),
    m_batch(bats),
    m_sequenceLength(sequenceLength)
{
    NNHardAssertEquals(feat.dims(), 2, "Invalid features!");
    NNHardAssertEquals(lab.dims(), 2, "Invalid labels!");
    NNHardAssertEquals(feat.size(0), lab.size(0), "Incompatible features and labels!");
    NNHardAssertLessThanOrEquals(sequenceLength, feat.size(0), "Invalid sequence length!");
    NNHardAssertLessThanOrEquals(bats, feat.size(0), "Invalid batch size!");
    reset();
}

template <typename T>
SequenceBatcher<T> &SequenceBatcher<T>::sequenceLength(size_t sequenceLength)
{
    NNAssertLessThanOrEquals(sequenceLength, m_feat.size(0), "Invalid sequence length!");
    m_sequenceLength = sequenceLength;
    m_featBatch.resizeDim(0, sequenceLength);
    m_labBatch.resizeDim(0, sequenceLength);
    reset();
    return *this;
}

template <typename T>
size_t SequenceBatcher<T>::sequenceLength() const
{
    return m_sequenceLength;
}

template <typename T>
SequenceBatcher<T> &SequenceBatcher<T>::batch(size_t bats)
{
    NNAssertLessThanOrEquals(bats, m_feat.size(0), "Invalid batch size!");
    m_batch = bats;
    m_featBatch.resizeDim(1, bats);
    m_labBatch.resizeDim(1, bats);
    reset();
    return *this;
}

template <typename T>
size_t SequenceBatcher<T>::batch() const
{
    return m_batch;
}

template <typename T>
SequenceBatcher<T> &SequenceBatcher<T>::reset()
{
    Storage<size_t> indices(m_batch);
    for(size_t &index : indices)
    {
        index = Random<size_t>::sharedRandom().uniform(m_feat.size(0) - m_sequenceLength + 1);
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

template <typename T>
Tensor<T> &SequenceBatcher<T>::features()
{
    return m_featBatch;
}

template <typename T>
Tensor<T> &SequenceBatcher<T>::labels()
{
    return m_labBatch;
}

template <typename T>
const Tensor<T> &SequenceBatcher<T>::allFeatures() const
{
    return m_featBatch;
}

template <typename T>
const Tensor<T> &SequenceBatcher<T>::allLabels() const
{
    return m_labBatch;
}

}

#endif
