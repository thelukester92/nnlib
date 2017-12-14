#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/critics/criticsequencer.hpp"
#include "nnlib/critics/detail/criticsequencer.tpp"

template class nnlib::CriticSequencer<NN_REAL_T>;

#endif
