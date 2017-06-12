#ifndef TEST_RMSPROP_H
#define TEST_RMSPROP_H

#include "nnlib/opt/rmsprop.h"
#include "nnlib/nn/linear.h"
#include "nnlib/critics/mse.h"
using namespace nnlib;

void TestRMSProp()
{
	Linear<> nn(2, 3);
	MSE<> critic(nn.outputs());
	
	RMSProp<> opt(nn, critic);
	/// \todo fill me in
}

#endif
