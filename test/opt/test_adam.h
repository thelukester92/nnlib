#ifndef TEST_ADAM_H
#define TEST_ADAM_H

#include "nnlib/opt/adam.h"
#include "nnlib/nn/linear.h"
#include "nnlib/critics/mse.h"
using namespace nnlib;

void TestAdam()
{
	Linear<> nn(2, 3);
	MSE<> critic(nn.outputs());
	
	Adam<> opt(nn, critic);
	/// \todo fill me in
}

#endif
