#ifndef UTIL_BATCHER_HPP
#define UTIL_BATCHER_HPP

#include "../math/random.hpp"
#include "../core/tensor.hpp"

namespace nnlib
{

/// Takes two tensors and returns random slices along the major dimensions, one slice at at time.
/// This is useful for optimization.
/// Batcher requires non-const inputs and will shuffle them unless the copy flag is true.
template <typename T = NN_REAL_T>
class Batcher
{
public:
    Batcher(Tensor<T> &feat, Tensor<T> &lab, size_t bats = 1, bool copy = false);
    Batcher(const Tensor<T> &feat, const Tensor<T> &lab, size_t bats);

    Batcher &batch(size_t bats);
    size_t batch() const;
    size_t batches() const;

    Batcher &reset();

    bool next(bool autoReset = false);

    Tensor<T> &features();
    Tensor<T> &labels();
    Tensor<T> &allFeatures();
    Tensor<T> &allLabels();

private:
    Tensor<T> m_feat;
    Tensor<T> m_lab;
    Tensor<T> m_featBatch;
    Tensor<T> m_labBatch;
    size_t m_offset;
    size_t m_batch;
};

/// This variation of Batcher yields sequences of batches (for sequential data).
/// Unlike the regular Batcher, the SequenceBatcher only yields one sequence per reset,
/// so there is no "next" method.
template <typename T = NN_REAL_T>
class SequenceBatcher
{
public:
    SequenceBatcher(Tensor<T> &&feat, Tensor<T> &&lab, size_t sequenceLength = 1, size_t bats = 1);
    SequenceBatcher(const Tensor<T> &feat, const Tensor<T> &lab, size_t sequenceLength = 1, size_t bats = 1);

    SequenceBatcher &sequenceLength(size_t sequenceLength);
    size_t sequenceLength() const;

    SequenceBatcher &batch(size_t bats);
    size_t batch() const;

    SequenceBatcher &reset();

    Tensor<T> &features();
    Tensor<T> &labels();

private:
    Tensor<T> m_feat;
    Tensor<T> m_lab;
    Tensor<T> m_featBatch;
    Tensor<T> m_labBatch;
    size_t m_batch;
    size_t m_sequenceLength;
};

}

#if defined NN_REAL_T && !defined NN_IMPL
    extern template class nnlib::Batcher<NN_REAL_T>;
    extern template class nnlib::SequenceBatcher<NN_REAL_T>;
#elif !defined NN_IMPL
    #include "detail/batcher.tpp"
#endif

#endif
