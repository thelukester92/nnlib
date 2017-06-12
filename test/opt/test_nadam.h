#ifndef TEST_NADAM_H
#define TEST_NADAM_H

#include "nnlib/opt/nadam.h"
#include "nnlib/nn/linear.h"
#include "nnlib/critics/mse.h"
using namespace nnlib;

void TestNadam()
{
	Linear<> nn(2, 3);
	MSE<> critic(nn.outputs());
	
	Nadam<> opt(nn, critic);
	/// \todo fill me in
}

#endif
