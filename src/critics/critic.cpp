#ifdef NN_REAL_T
#define NN_IMPL

#include "nnlib/critics/critic.hpp"
#include "nnlib/critics/detail/critic.tpp"

template class nnlib::Critic<NN_REAL_T>;

#endif
