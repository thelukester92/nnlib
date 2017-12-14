#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/opt/rmsprop.hpp"
#include "nnlib/opt/detail/rmsprop.tpp"

template class nnlib::RMSProp<NN_REAL_T>;

#endif
