#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/util/batcher.hpp"
#include "nnlib/util/detail/batcher.tpp"

template class nnlib::Batcher<NN_REAL_T>;
template class nnlib::SequenceBatcher<NN_REAL_T>;

#endif
