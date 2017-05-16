#include "nnlib/nn/batchnorm.h"
using namespace nnlib;

void TestBatchNorm()
{
	Tensor<> inp = Tensor<>({
		 3,  6,  9,
		-1,  5,  4,
		12,  5, 11
	}).reshape(3, 3);
	
	BatchNorm<> bn(3, 3);
	bn.forward(inp);
	
	for(size_t i = 0; i < 3; ++i)
	{
		NNHardAssert(fabs(bn.output().select(1, i).mean()) < 1e-9, "Non-zero mean!");
		NNHardAssert(fabs(bn.output().select(1, i).variance() - 1) < 1e-9, "Non-unit variance!");
	}
}
