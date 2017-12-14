#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/critics/nll.hpp"
#include "nnlib/critics/detail/nll.tpp"

template class nnlib::NLL<NN_REAL_T>;

#endif
