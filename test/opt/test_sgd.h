#ifndef TEST_SGD_H
#define TEST_SGD_H

#include "nnlib/opt/sgd.h"
#include "nnlib/nn/linear.h"
#include "nnlib/critics/mse.h"
using namespace nnlib;

void TestSGD()
{
	Linear<> nn(2, 3);
	MSE<> critic(nn.outputs());
	
	SGD<> opt(nn, critic);
	/// \todo fill me in
}

#endif
